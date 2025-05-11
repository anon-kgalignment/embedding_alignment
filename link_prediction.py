import torch
import numpy as np
from tqdm import tqdm
from dicee import KGE
from dicee.static_funcs import get_er_vocab, get_re_vocab

@torch.no_grad()
def evaluate_link_prediction_performance(model: KGE, triples, er_vocab, re_vocab):
    assert isinstance(model, KGE)
    model.model.eval()
    hits = dict()
    reciprocal_ranks = []
    num_entities = model.num_entities
    all_entities = torch.arange(0, num_entities).long()

    skipped = 0  # Count how many triples were skipped

    for i in tqdm(range(len(triples))):
        str_h, str_r, str_t = triples[i]

        try:
            h = model.get_entity_index(str_h)
            r = model.get_relation_index(str_r)
            t = model.get_entity_index(str_t)
        except KeyError:
            skipped += 1
            continue  # Skip this triple if any part is not found

        x_tail = torch.stack((torch.tensor(h).repeat(num_entities),
                              torch.tensor(r).repeat(num_entities),
                              all_entities), dim=1)
        predictions_tails = model.model.forward_triples(x_tail)

        x_head = torch.stack((all_entities,
                              torch.tensor(r).repeat(num_entities),
                              torch.tensor(t).repeat(num_entities)), dim=1)
        predictions_heads = model.model.forward_triples(x_head)

        filt_tails = [model.entity_to_idx[i] for i in er_vocab.get((str_h, str_r), []) if i in model.entity_to_idx]
        target_tail_score = predictions_tails[t].item()
        predictions_tails[filt_tails] = -np.Inf
        predictions_tails[t] = target_tail_score
        rank_t = (predictions_tails.argsort(descending=True) == t).nonzero(as_tuple=True)[0].item() + 1

        filt_heads = [model.entity_to_idx[i] for i in re_vocab.get((str_r, str_t), []) if i in model.entity_to_idx]
        target_head_score = predictions_heads[h].item()
        predictions_heads[filt_heads] = -np.Inf
        predictions_heads[h] = target_head_score
        rank_h = (predictions_heads.argsort(descending=True) == h).nonzero(as_tuple=True)[0].item() + 1

        reciprocal_ranks.append(1 / rank_t + 1 / rank_h)
        for k in [1, 3, 10]:
            hits.setdefault(k, []).append((rank_t <= k) + (rank_h <= k))

    total_evaluated = len(triples) - skipped
    if total_evaluated == 0:
        print("No valid triples were evaluated.")
        return {"H@1": 0.0, "H@3": 0.0, "H@10": 0.0, "MRR": 0.0}

    mrr = sum(reciprocal_ranks) / (2 * total_evaluated)
    results = {
        "H@1": sum(hits[1]) / (2 * total_evaluated),
        "H@3": sum(hits[3]) / (2 * total_evaluated),
        "H@10": sum(hits[10]) / (2 * total_evaluated),
        "MRR": mrr
    }

    print(f"Skipped {skipped} triples due to missing entities/relations.")
    return results


def evaluate_model(model: KGE, test_triples):
    if model is None or test_triples is None:
        return -1
    er_vocab = get_er_vocab(test_triples)
    re_vocab = get_re_vocab(test_triples)
    return evaluate_link_prediction_performance(model, test_triples, er_vocab, re_vocab)
