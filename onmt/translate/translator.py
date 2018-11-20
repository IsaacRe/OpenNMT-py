#!/usr/bin/env python
from __future__ import print_function
"""
Translator Class and builder

Modified for Version 0 functionality
Changes tagged with 'V0 Modification'

Modified for Version 1 functionality
Changes tagged with 'V1 Modification'
"""
import argparse
import codecs
import os
import math

import torch

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.opts as opts
import onmt.decoders.ensemble
from onmt.utils.knowledge_sink import StateUpdater  # V1 Modification: import StateUpdater - Isaac
import contextlib  # V1 Modification - Isaac


# V1 Modification: pass knowledge_sink object to translator contruction
def build_translator(opt, report_score=True, logger=None, out_file=None, knowledge_sink=None):

    # V1 Modification: make use of gpu option (vanilla codebase doesn't)

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    # End Modification

    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    # V0 Modification: add filewrite stream to save ground truth completions - Isaac

    out_gt_file = None
    out_gt_filename = None
    out_hint_file = None
    out_hint_filename = None

    # V1 Modification: add filewrite stream to save update completions - Isaac

    out_update_file = None
    out_update_filename = None
    if opt.num_gt > 0:
        out_gt_filename = '.'.join(opt.output.split('.')[:-1] + ['gt'] + [opt.output.split('.')[-1]])
        out_gt_file = codecs.open(out_gt_filename, 'w+', 'utf-8')
        if opt.update_size > 0:
            out_update_filename = '.'.join(opt.output.split('.')[:-1] + ['update'] + [opt.output.split('.')[-1]])
            out_update_file = codecs.open(out_update_filename, 'w+', 'utf-8')
        else:
            out_hint_filename = '.'.join(opt.output.split('.')[:-1] + ['hint'] + [opt.output.split('.')[-1]])
            out_hint_file = codecs.open(out_hint_filename, 'w+', 'utf-8')

    # End Modification

    # End Modification

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    if len(opt.models) > 1:
        # use ensemble decoding if more than one model is specified
        fields, model, model_opt = \
            onmt.decoders.ensemble.load_test_model(opt, dummy_opt.__dict__)
    else:
        fields, model, model_opt = \
            onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        "sample_rate", "window_size", "window_stride",
                        "window", "num_gt", "image_channel_size"]}  # V0 Modification: add num_gt to args - Isaac

    translator = Translator(model, fields, global_scorer=scorer, out_file=out_file,

                            # V0 Modification: pass added constructor args - Isaac

                            out_gt_file=out_gt_file,
                            out_gt_filename=out_gt_filename,
                            out_hint_file=out_hint_file,
                            out_hint_filename=out_hint_filename,

                            # End Modification

                            # V1 Modification - Isaac

                            out_update_file=out_update_file,
                            out_update_filename=out_update_filename,

                            # End Modification

                            report_score=report_score,
                            copy_attn=model_opt.copy_attn,
                            logger=logger,
                            **kwargs)

    # V1 Modification: attach knowledge_sink to translator

    if opt.update_size > 0 and opt.num_gt > 0:
        translator.knowledge_sink = StateUpdater(opt.update_size)

    # End Modification

    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,

                 # V0 Modification: added constructor args - Isaac

                 out_gt_file=None,
                 out_gt_filename=None,
                 out_hint_file=None,
                 out_hint_filename=None,

                 # End Modification

                 # V1 Modification - Isaac

                 out_update_file=None,
                 out_update_filename=None,

                 # End Modification

                 fast=False,
                 num_gt=0,  # V0 Modification: add number of hints as translator arg - Isaac
                 image_channel_size=3):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file

        # V0 Modification: save ground truth completions to separate file, save num_gt as member var - Isaac

        self.num_gt = num_gt
        self.out_gt_file = out_gt_file
        self.out_gt_filename = out_gt_filename
        self.out_hint_file = out_hint_file
        self.out_hint_filename = out_hint_filename

        # End Modification

        # V1 Modification - Isaac

        self.knowledge_sink = None
        self.out_update_file = out_update_file
        self.out_update_filename = out_update_filename

        # End Modification

        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast
        self.image_channel_size = image_channel_size

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self,
                  src_path=None,
                  src_data_iter=None,
                  tgt_path=None,
                  tgt_data_iter=None,
                  src_dir=None,
                  batch_size=None,
                  baseline=None,  # V0 Modification: add boolean arg to allow sampling of completion comparisons - Isaac
                  attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            src_data_iter (iterator): an interator generating source data
                e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            tgt_data_iter (iterator): an interator generating target data
            src_dir (str): source directory path
                (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            baseline (bool): whether to ignore ground truth and follow baseline sampling to get a baseline for
                            completions sampled with ground truth hints. If True, will additionally skip writing
                            of ground truth completions to separate output file
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        # V0, V1 Modification: Check for target data - Isaac

        num_gt = self.num_gt
        update = self.knowledge_sink and not baseline

        # If we are testing performance with hints make sure we have access to ground truth translations
        if num_gt and num_gt > 0 and not baseline:
            assert tgt_data_iter is not None or tgt_path is not None

        # End Modification

        if batch_size is None:
            raise ValueError("batch_size must be set")
        data = inputters. \
            build_dataset(self.fields,
                          self.data_type,
                          src_path=src_path,
                          src_data_iter=src_data_iter,
                          tgt_path=tgt_path,
                          tgt_data_iter=tgt_data_iter,
                          src_dir=src_dir,
                          sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          use_filter_pred=self.use_filter_pred,
                          image_channel_size=self.image_channel_size)

        if self.cuda:
            cur_device = "cuda"
        else:
            cur_device = "cpu"

        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path, num_gt=num_gt)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        for batch in data_iter:

            # V0 Modification: pass number of hints to translate_batch - Isaac

            # set num_gt to 0 to specify translation without hint if baseline is true
            batch_data = self.translate_batch(batch, data, fast=self.fast,
                                              num_gt=0 if baseline and not self.knowledge_sink else num_gt,
                                              update=update)

            # End Modification

            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                # V0 Modification: only write completion to out file - Isaac

                n_best_preds = [" ".join(pred[num_gt:])
                                for pred in trans.pred_sents[:self.n_best]]

                # End Modification

                all_predictions += [n_best_preds]

                # V0, V1 Modification: write ground truth completions to the specified, separate out file

                # only write to ground truth file when following hint sampling (avoid redundancy)
                if num_gt > 0 and not baseline:
                    self.out_gt_file.write(' '.join(trans.gold_sent[num_gt:]) + '\n')
                    self.out_gt_file.flush()
                    if update:
                        file_stream = self.out_update_file
                    else:
                        file_stream = self.out_hint_file
                else:
                    if update:
                        file_stream = self.out_hint_file
                    else:
                        file_stream = self.out_file
                file_stream.write('\n'.join(n_best_preds) + '\n')
                file_stream.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    if self.data_type == 'text':
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *srcs) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            if tgt_path is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                if self.report_bleu:

                    # V0 Modification: provide ground truth completions path to scorer - Isaac

                    msg = self._report_bleu(tgt_path if num_gt == 0 else self.out_gt_filename, file_stream)

                    # End Modification

                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                if self.report_rouge:

                    # V0 Modification: provide ground truth completions path to scorer - Isaac

                    msg = self._report_rouge(tgt_path if num_gt == 0 else self.out_gt_filename, file_stream)

                    # End Modification

                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores, all_predictions

    def translate_batch(self, batch, data, fast=False, num_gt=0, update=False):  # V0, V1 Modification: add args - Isaac
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)
           num_gt (int): the number of ground truth hints to provide during translation
           update (bool):  whether to perform an update on the cell state or not

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad() if not update else contextlib.suppress():  # V1 Modification: if updating retain gradient
                                                                        # functionality
            if fast:
                return self._fast_translate_batch(
                    batch,
                    data,
                    self.max_length,
                    min_length=self.min_length,
                    n_best=self.n_best,
                    return_attention=self.replace_unk)
            else:
                return self._translate_batch(batch, data, num_gt=num_gt, update=update)  # V0 Modification: add args
                                                                                         # - Isaac

    # V1 Modification: fast_forward function to get logits deeper method returns to compute loss

    def fast_forward(self, dec_out, dec_states, attn, unbottle, data, src_map, batch, beam_size):
        dec_out = dec_out.squeeze(0)

        # dec_out: beam x rnn_size

        # (b) Compute a vector of batch x beam word scores.
        if not self.copy_attn:
            # out will not be used since ground truth will be next step's input, so no need for further processing
            """
            out = self.model.generator.forward(dec_out).data
            out = unbottle(out)
            # beam x tgt_vocab
            """
            beam_attn = unbottle(attn["std"].repeat(beam_size, 1, 1))
        else:
            # out will not be used since ground truth will be next step's input, so no need for further processing
            """
            out = self.model.generator.forward(dec_out,
                                               attn["copy"].squeeze(0),
                                               src_map)
            # beam x (tgt_vocab + extra_vocab)
            out = data.collapse_copy_scores(
                unbottle(out.data),
                batch, self.fields["tgt"].vocab, data.src_vocabs)
            # beam x tgt_vocab
            out = out.log()
            """
            beam_attn = unbottle(attn["copy"].repeat(beam_size, 1, 1))
        out = None

        return dec_out, dec_states, attn, out, beam_attn

    # End Modification

    def _fast_translate_batch(self,
                              batch,
                              data,
                              max_length,
                              min_length=0,
                              n_best=1,
                              return_attention=False):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert data.data_type == 'text'
        assert not self.copy_attn
        assert not self.dump_beam
        assert not self.use_filter_pred
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0

        beam_size = self.beam_size
        batch_size = batch.batch_size
        vocab = self.fields["tgt"].vocab
        start_token = vocab.stoi[inputters.BOS_WORD]
        end_token = vocab.stoi[inputters.EOS_WORD]

        # Encoder forward.
        src = inputters.make_features(batch, 'src', data.data_type)
        _, src_lengths = batch.src
        enc_states, memory_bank, src_lengths \
            = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=memory_bank.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=memory_bank.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=memory_bank.device)
        alive_attn = None

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=memory_bank.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)

            # Decoder forward.
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_lengths=memory_lengths,
                step=step)

            # Generator forward.
            log_probs = self.model.generator.forward(dec_out.squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                attention = (
                    alive_attn.view(
                        alive_attn.size(0), -1, beam_size, alive_attn.size(-1))
                    if alive_attn is not None else None)
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_lengths[i]]
                            if attention is not None else None))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                        .view(alive_attn.size(0),
                              -1, alive_attn.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_lengths = memory_lengths.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

    def _translate_batch(self, batch, data, num_gt=0, update=False):  # V0, V1 Modification: add num_gt argument - Isaac
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[inputters.PAD_WORD],
                                    eos=vocab.stoi[inputters.EOS_WORD],
                                    bos=vocab.stoi[inputters.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a, grad=False):
            return torch.tensor(a, requires_grad=grad)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # V0 Modification - Isaac

        # Prepare ground truth
        if num_gt > 0:
            # add 1 since first symbol is <start>
            ground_truth = inputters.make_features(batch, 'tgt', 'text')[:num_gt+1, :, 0]  # [num_gt X batch]

        # End Modification

        # (1) Run the encoder on the src.
        src = inputters.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        enc_states, memory_bank, src_lengths \
            = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)

        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data) \
                .long() \
                .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None

        # V0 Modification: don't expand memory and dec_states by beam_size until done processing ground truth

        """
        if isinstance(memory_bank, tuple):
            memory_bank_d = tuple(rvar(x.data) for x in memory_bank)
        else:
            memory_bank_d = rvar(memory_bank.data)
        memory_lengths_d = src_lengths.repeat(beam_size)
        dec_states_d = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)
        dec_states_d.repeat_beam_size_times(beam_size)
        """
        memory_lengths = src_lengths

        # End Modification

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # V0 Modification: if done processing ground truth, expand decoder inputs along batch dim by beam size

            if i == ground_truth.size(0):
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(rvar(x.data) for x in memory_bank)
                else:
                    memory_bank = rvar(memory_bank.data)
                memory_lengths = memory_lengths.repeat(beam_size)
                dec_states.repeat_beam_size_times(beam_size)

            # End Modification

            # V0 Modification: set previous ground truth as input to each step - Isaac

            if num_gt > 0 and i < ground_truth.size(0):
                # fill inp with ground truth words
                inp = ground_truth[i].unsqueeze(0)
            else:
                # Construct batch x beam_size nxt words.
                # Get all the pending current beam words and arrange for forward.
                inp = var(torch.stack([b.get_current_state() for b in beam])
                          .t().contiguous().view(1, -1))

            # End Modification

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # V1 Modification: save variables for call to fast_forward - Isaac

            if i + 1 == num_gt and update:
                self.knowledge_sink.save_vars(self,
                                              unbottle=unbottle,
                                              data=data,
                                              src_map=src_map,
                                              batch=batch,
                                              beam_size=beam_size)

            # End Modification

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, memory_bank, dec_states,
                memory_lengths=memory_lengths,
                step=i,
                knowledge_sink=self.knowledge_sink if update and i + 1 == num_gt else None)  # V1 Modification:
                                                                                             # pass knowledge sink to
                                                                                             # decoder if we're doing
                                                                                             # update this step

            dec_out = dec_out.squeeze(0)

            # V0 Modification: if we are processing ground truth we need to expand attn dims for beam size

            if i < ground_truth.size(0):
                for k in attn:
                    attn[k] = attn[k].repeat(1, beam_size, 1)

            # End Modification

            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:

                # V0 Modification: only process output into word predictions if we are not feeding ground truth
                # or are updating - Isaac

                if i >= num_gt or (update and i + 1 == num_gt):
                    out = self.model.generator.forward(dec_out)
                    # if last step of ground truth input, we need to expand dim
                    if i == num_gt:
                        out = out.repeat(beam_size, 1)
                    # don't unbottle during update step
                    if i >= num_gt:
                        out = unbottle(out)
                if i <= num_gt:
                    attn["std"] = attn["std"].repeat(beam_size, 1, 1)

                # End Modification

                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:

                # V0 Modification: only process output into word predictions if we are not feeding ground truth
                # or are updating - Isaac

                if i >= num_gt or (update and i + 1 == num_gt):
                    out = self.model.generator.forward(dec_out,
                                                       attn["copy"].squeeze(0),
                                                       src_map)
                    # if last step of ground truth input, we need to expand dim
                    if i == num_gt:
                        out = out.repeat(beam_size, 1)
                        # beam x (tgt_vocab + extra_vocab)
                    # don't unbottle during update step
                    if i >= num_gt:
                        out = data.collapse_copy_scores(
                            unbottle(out),
                            batch, self.fields["tgt"].vocab, data.src_vocabs)
                        # beam x tgt_vocab
                        out = out.log()
                if i <= num_gt:
                    attn["copy"] = attn["copy"].repeat(beam_size, 1, 1)

                # End Modification

                beam_attn = unbottle(attn["copy"])

            # End Modification

            # V1 Modification: conduct knowledge injection - Isaac

            # currently only perform update during final step for which we have ground truth
            if i + 1 == num_gt and update:
                dec_out, dec_states, attn, out, beam_attn = self.knowledge_sink.correct(out, ground_truth[i+1])

            # End Modification

            # V0 Modification - Isaac

            # output is log-probs over vocab, so set ground truth index to 0, all others to be very negative
            def one_hot_log(idxs, cuda):
                if cuda:
                    return torch.cuda.FloatTensor([[0.0 if x == idx else float("-inf") for x in range(len(vocab))] for idx in idxs]).repeat(beam_size, 1, 1)
                return torch.FloatTensor([[0.0 if x == idx else float("-inf") for x in range(len(vocab))] for idx in idxs]).repeat(beam_size, 1, 1)

            if i < num_gt:
                out = one_hot_log(ground_truth[i+1], self.cuda)

            # End Modification

            # V0 Modification: if final step of ground input, expand outputs by beam size

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]],
                          i < num_gt)  # V0 Modification: pass parameter specifying whether output is ground truth
                                       # - Isaac
                # V0 Modification: only need to update decoder states if currently running beam search

                if i >= ground_truth.size(0):
                    dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        elif data_type == 'audio':
            src_lengths = batch.src_lengths
        else:
            src_lengths = None
        src = inputters.make_features(batch, 'src', data_type)
        tgt_in = inputters.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank, src_lengths \
            = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[inputters.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores.view(-1)
        return gold_scores

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    # V0 Modification: app optional arg specifying predictions stream reader - Isaac

    def _report_bleu(self, tgt_path, hyp_stream=None):
        if not hyp_stream:
            hyp_stream = self.out_file

    # End Modification

        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (base_dir, tgt_path),
                                      stdin=hyp_stream,  # V0 Modification: set stdin=hyp_stream - Isaac
                                      shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    # V0 Modification: app optional arg specifying predictions stream reader - Isaac

    def _report_rouge(self, tgt_path, hyp_stream=None):
        if not hyp_stream:
            hyp_stream = self.out_file

    # End Modification

        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=hyp_stream).decode("utf-8")  # V0 Modification: set stdin=hyp_stream - Isaac
        msg = res.strip()
        return msg
