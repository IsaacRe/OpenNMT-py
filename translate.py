#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
"""
Modified for Version 0 functionality
Changes tagged with 'V0 Modification'
"""

import argparse
import traceback, sys, pdb

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts


def main(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)

    # V0 Modification: translate a second time to get baseline completion for comparison - Isaac

    if opt.num_gt > 0:

        translator.translate(src_path=opt.src,
                             tgt_path=opt.tgt,
                             src_dir=opt.src_dir,
                             batch_size=opt.batch_size,
                             attn_debug=opt.attn_debug,
                             baseline=True)  # specify baseline sampling procedure (no hint)

    # End Modification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    """
    try:
        main(opt)
    except:
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    """
    main(opt)
