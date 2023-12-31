from typing import Dict, List, Tuple

import numpy
import numpy as np
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from torch.autograd import Variable


@Model.register("disentangle_seq2seq")
class DisentangleSeq2Seq(Model):


    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 syntax_encoder: Seq2SeqEncoder,
                 semantic_encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 z_dim: int = 500,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 beam_size: int = None,
                 target_embedder: TextFieldEmbedder = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 use_bleu: bool = True) -> None:
        super(DisentangleSeq2Seq, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
            self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})
        else:
            self._bleu = None

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        beam_size = beam_size or 1
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._syntax_encoder = syntax_encoder
        self._semantic_encoder = semantic_encoder
        
        self.z_dim = z_dim
        self.q_mu = nn.Linear(syntax_encoder.get_output_dim(), z_dim)
        self.q_logvar = nn.Linear(syntax_encoder.get_output_dim(), z_dim)

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self._attention = attention
        elif attention_function:
            self._attention = LegacyAttention(attention_function)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        
        if (target_embedder):
            self._target_embedder = target_embedder
        else:
            self._target_embedder = Embedding(num_classes, target_embedding_dim)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._syntax_encoder_output_dim = z_dim#self._syntax_encoder.get_output_dim()
        self._semantic_encoder_output_dim = self._semantic_encoder.get_output_dim()
        self._decoder_output_dim = self._syntax_encoder_output_dim + self._semantic_encoder_output_dim

        self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        
        self._global_step = 0
        self.kld_start_inc = 3000
        
        self.max_kl_weight = 50
        self.kld_inc = self.max_kl_weight / (5000 - self.kld_start_inc)
        self.kl_weight = 0.0
        
        self.other_loss = {}

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.

        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    def get_mu_logvar(self, h):

        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar   
    
    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda()
        return mu + torch.exp(logvar/2) * eps
    
    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda()
        return z
    
    
    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens)
        syntax_state = {"source_mask" : state["source_mask"],
                        "encoder_outputs" : state["syntax_encoder_outputs"]}
        
        batch_size = state["source_mask"].size(0)
        
        # variational syntax_state
        syntax_h = syntax_state["encoder_outputs"]
        syntax_mu, syntax_logvar = self.get_mu_logvar(syntax_h)
        syntax_state["encoder_outputs"] = self.sample_z(syntax_mu, syntax_logvar)
        
        semantic_state = {"source_mask" : state["source_mask"],
                        "encoder_outputs" : state["semantic_encoder_outputs"]}
        
        
        target_state = self._encode(target_tokens)
        target_syntax_state = {"source_mask" : target_state["source_mask"],
                               "encoder_outputs" : target_state["syntax_encoder_outputs"]}
        # variational target_syntax_state
        target_syntax_h = target_syntax_state["encoder_outputs"]
        target_syntax_mu, target_syntax_logvar = self.get_mu_logvar(target_syntax_h)
        target_syntax_state["encoder_outputs"] = self.sample_z(target_syntax_mu, target_syntax_logvar)
        
        target_semantic_state = {"source_mask" : target_state["source_mask"],
                        "encoder_outputs" : target_state["semantic_encoder_outputs"]}

        #reconstruction
        recon_state = self._init_decoder_state(syntax_state, semantic_state)
        recon_output_dict = self._forward_loop(recon_state, source_tokens)
        
        if target_tokens:
            state = self._init_decoder_state(target_syntax_state, semantic_state)
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}
            
        #combined_output
        self.other_loss["transfer_loss"] = output_dict["loss"]
        
        output_dict["loss"] += recon_output_dict["loss"]
        
        self.other_loss["recon_loss"] = recon_output_dict["loss"]
        
        
        #KL loss
        self._global_step += 1
        
        if self._global_step > self.kld_start_inc:
            
            kl_loss = torch.mean(0.5 * torch.mean(torch.exp(syntax_logvar) + syntax_mu**2 - 1 - syntax_logvar, 1)) + torch.mean(0.5 * torch.mean(torch.exp(target_syntax_logvar) + target_syntax_mu**2 - 1 - target_syntax_logvar, 1))
            
            if (self.kl_weight < self.max_kl_weight):
                self.kl_weight += self.kld_inc
            else:
                self.kl_weight = self.max_kl_weight
                
            output_dict["loss"] += self.kl_weight * kl_loss
            
            self.other_loss["kl_loss"] = self.kl_weight * kl_loss
        else:
            self.other_loss["kl_loss"] = 0
            
        
        
        
        #semantic loss, current, triplet mse,   candidate loss F.cosine_similarity
        source_semantics = semantic_state["encoder_outputs"]
        positive_semantics = target_semantic_state["encoder_outputs"]
        negative_semantics = torch.cat([source_semantics[1:,:], source_semantics[0:1,:]], dim=0)
        diff_semantics = torch.nn.functional.mse_loss(source_semantics, positive_semantics) - torch.nn.functional.mse_loss(source_semantics, negative_semantics) + 1
        sem_loss = torch.where(diff_semantics>0, diff_semantics, torch.zeros_like(diff_semantics))
        output_dict["loss"] += sem_loss
        self.other_loss["sem_loss"] = sem_loss

        if not self.training:
            random_syntax_state = {"source_mask" : state["source_mask"],
                                   "encoder_outputs" : self.sample_z(syntax_mu, syntax_logvar)}#self.sample_z(syntax_mu, syntax_logvar) #self.sample_z_prior(batch_size)
            
            #random_syntax_state = #self._syntax_encode(target_tokens)#
            
            state = self._init_decoder_state(random_syntax_state, semantic_state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    
    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        syntax_encoder_outputs = self._syntax_encoder(embedded_input, source_mask)
        semantic_encoder_outputs = self._semantic_encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "syntax_encoder_outputs": syntax_encoder_outputs.mean(dim=1),
                "semantic_encoder_outputs": semantic_encoder_outputs.mean(dim=1)
        }
    
    def _syntax_encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._syntax_encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }
    
    def _semantic_encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._semantic_encoder(embedded_input, source_mask)
        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

#     def _init_decoder_state(self, syntax_state: Dict[str, torch.Tensor], semantic_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         batch_size = syntax_state["source_mask"].size(0)
#         syntax_final_encoder_output = util.get_final_encoder_states(
#                 syntax_state["encoder_outputs"],
#                 syntax_state["source_mask"],
#                 self._syntax_encoder.is_bidirectional())
#         semantic_final_encoder_output = util.get_final_encoder_states(
#                 semantic_state["encoder_outputs"],
#                 semantic_state["source_mask"],
#                 self._semantic_encoder.is_bidirectional())

        
#         state = {}
#         state["source_mask"] = semantic_state["source_mask"]
#         #print(syntax_state["encoder_outputs"].shape, semantic_state["encoder_outputs"])
#         state["syntax_encoder_outputs"] = syntax_state["encoder_outputs"]
#         state["semantic_encoder_outputs"] = semantic_state["encoder_outputs"]
#         state["decoder_hidden"] = torch.cat([syntax_final_encoder_output, semantic_final_encoder_output], dim=-1)
#         state["decoder_context"] = torch.cat([state["semantic_encoder_outputs"], state["semantic_encoder_outputs"]], dim=-1).new_zeros(batch_size, self._decoder_output_dim)
#         return state

    def _init_decoder_state(self, syntax_state: Dict[str, torch.Tensor], semantic_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #print(syntax_state["source_mask"].shape)
        batch_size = syntax_state["source_mask"].size(0)
        
        state = {}
        state["source_mask"] = semantic_state["source_mask"]
        #print(syntax_state["encoder_outputs"].shape, semantic_state["encoder_outputs"])
        state["syntax_encoder_outputs"] = syntax_state["encoder_outputs"]
        state["semantic_encoder_outputs"] = semantic_state["encoder_outputs"]
        state["decoder_hidden"] = torch.cat([state["syntax_encoder_outputs"], state["semantic_encoder_outputs"]], dim=-1)
        state["decoder_context"] = state["semantic_encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)
        return state

    

    
    
    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.

        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.

        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        syntax_encoder_outputs = state["syntax_encoder_outputs"]
        semantic_encoder_outputs = state["semantic_encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)#tobedone

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            input_weights = torch.softmax(source_mask.float(), dim=-1)
            #print(encoder_outputs.shape, input_weights.shape)
            attended_input = util.weighted_sum(encoder_outputs, input_weights)

            decoder_input = torch.cat((attended_input, embedded_input), -1)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

    def _prepare_attended_input(self,
                                decoder_hidden_state: torch.LongTensor = None,
                                encoder_outputs: torch.LongTensor = None,
                                encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.

        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        
        all_metrics.update(self.other_loss)
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
