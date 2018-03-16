#!/usr/bin/env python

import argparse
import itertools
import os
import sys


def main(args):
    with open(args.truth) as truth, open(args.predictions) as predictions:
        nb_matches = 0
        nb_total = 0

        for t, p in itertools.izip_longest(truth,predictions, fillvalue='skip'):
            if t == 'skip' or p == 'skip':
                print 'mismatch for', t, p, ' - skipping'
                break

            t_image, t_label = map(str.strip, t.split())
            p_image, p_label = map(str.strip, p.split())

            if t_image != p_image:
                print 'image mismatch, please check that files are aligned', t_image, p_image
                continue
            
            nb_total += 1
            nb_matches += 1 if t_label == p_label else 0

            if args.verbose and t_label != p_label:
                print 'label mismatch for', t_image, ':', t_label, p_label

        accuracy = float(nb_matches) / nb_total

        print '\n'
        print 'number of correct matches', nb_matches,
        print 'for', nb_total, 'images'
        print 'accuracy', accuracy



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'validate classification predictions line by line'
        ' in format "<image> <label>'
    ));
    parser.add_argument('--truth', help='path to file containing ground truth', required=True)
    parser.add_argument('--predictions', help='path to file containing predictions', required=True)
    parser.add_argument('--verbose',
                        action='store_true',
                        help='report labels which doesn\'t match', 
                        default=False)
    args = parser.parse_args()
    main(args)

