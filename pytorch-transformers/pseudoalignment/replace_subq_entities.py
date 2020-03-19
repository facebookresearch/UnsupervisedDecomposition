import argparse
import os
import random
import spacy
from tqdm import tqdm


ENTLABELS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW',
             'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'ANY']


def get_entlabel2ents(q_tok, seeded_random):
    q_entlabel2ents = {label: [] for label in ENTLABELS}
    for ent in q_tok.ents:
        q_entlabel2ents[ent.label_].append(ent.text.lower())
        q_entlabel2ents['ANY'].append(ent.text.lower())
    for label in ENTLABELS:
        seeded_random.shuffle(q_entlabel2ents[label])
    return q_entlabel2ents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default='valid',  # E.g., 'train', 'valid', 'test', or 'train.04' (sharded)
                        type=str, help="Data split to run on.")
    parser.add_argument("--debug", action='store_true', help="Use debug mode?")
    parser.add_argument("--replace_by_type", action='store_true', help="Replace all entities (regardless of type)?")
    parser.add_argument("--data_folder", required=True,
                        type=str, help="Data to run on. E.g., all.paired_two_step_nn.min_q_len=4")
    args = parser.parse_args()

    seeded_random = random.Random(42)
    nlp = spacy.load("en_core_web_lg")

    # Read data
    umt_data_dir = f'data/umt/{args.data_folder}'
    with open(f'{umt_data_dir}/{args.split}.mh') as f:
        qs = f.readlines()
    qs = [q.strip('\n') for q in qs]
    with open(f'{umt_data_dir}/{args.split}.sh') as f:
        sqs = f.readlines()
    sqs = [sq.strip('\n') for sq in sqs]

    # Replace sub-q entities with q entities
    num_sqs_changed = 0
    new_sqs = []
    for q_raw, sq_raw in tqdm(zip(qs, sqs), total=len(qs)):
        q_entlabel2ents = get_entlabel2ents(nlp(q_raw), seeded_random)
        sq_entlabel2ents = get_entlabel2ents(nlp(sq_raw), seeded_random)
        q_raw_lower = q_raw.lower()
        sq_new = sq_raw.lower()
        sq_changed = False
        for sq_entlabel, sq_entlist in sq_entlabel2ents.items():
            if (len(sq_entlist) == 0) or (sq_entlabel == 'ANY'):
                continue
            q_entlabel = 'ANY' if (not args.replace_by_type) and (len(q_entlabel2ents[sq_entlabel]) == 0) else sq_entlabel
            q_entlist = q_entlabel2ents[q_entlabel]
            seeded_random.shuffle(q_entlist)
            if len(q_entlist) > 0:
                new_sq_ent_no = 0
                for sq_ent in sq_entlist:
                    if sq_ent in q_raw_lower:
                        continue
                    q_ent = q_entlist[new_sq_ent_no % len(q_entlist)]
                    new_sq_ent_no += 1
                    sq_new_updated = sq_new.replace(sq_ent, q_ent)
                    if sq_new_updated != sq_new:
                        sq_changed = True
                    sq_new = sq_new_updated
        num_sqs_changed += sq_changed
        new_sqs.append(sq_new)
        if args.debug and (num_sqs_changed > 5):
            break

    print(f'Num SQs changed: {num_sqs_changed}')
    save_dir = f'{umt_data_dir}.replace_entity_{"all" if not args.replace_by_type else "by_type"}'
    print(f'Saving to {save_dir}...')
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{args.split}.sh', 'w') as f:
        f.writelines('\n'.join(new_sqs) + '\n')
    print(f'Done! Saved to {save_dir}/{args.split}.sh')


if __name__ == "__main__":
    main()
