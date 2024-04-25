import numpy as np
import itertools
import ccobra
import sys
import json

A = "A"
B = "B"
C = "C"

all_terms = [A, B, C]

def int_to_bools(num, digits):
    return [bool(num & (1<<n)) for n in range(digits)]

def get_all_expanded_sets(initial, subj, obj, terms):
    results = []
    remaining_terms = np.array([x for x in terms if x != subj and x != obj])
    
    n_terms = len(remaining_terms)
    for i in range(2**n_terms):
        mask = int_to_bools(i, n_terms)
        to_append = remaining_terms[mask]
        results.append(sorted(initial + to_append.tolist()))
    
    return results
    
def get_premise_meanings(quant, subj, obj):
    positives = []
    negatives = []
    
    if quant == "A":
        # all subj are obj
        positives.append(get_all_expanded_sets([subj, obj], subj, obj, all_terms))
        negatives.extend(get_all_expanded_sets([subj], subj, obj, all_terms))
    elif quant == "E":
        # no subj are obj
        positives.append(get_all_expanded_sets([subj], subj, obj, all_terms))
        positives.append(get_all_expanded_sets([obj], subj, obj, all_terms))
        negatives.extend(get_all_expanded_sets([subj, obj], subj, obj, all_terms))
    elif quant == "I":
        # some subj are obj
        positives.append(get_all_expanded_sets([subj, obj], subj, obj, all_terms))
    elif quant == "O":
        # some subj are not obj
        positives.append(get_all_expanded_sets([subj], subj, obj, all_terms))
    # TODO: extend to generalized quantifiers
    
    return (positives, negatives)

def all_combinations(worlds):
    tupled_worlds = []
    for world in worlds:
        tupled_worlds.append([tuple(x) for x in world])
    
    combinations = [set(c) for c in itertools.product(*tupled_worlds)]
    return combinations


def create_world_set(syl):
    quant1 = syl[0]
    quant2 = syl[1]
    figure = int(syl[2])
    
    subj1 = None
    subj2 = None
    obj1 = None
    obj2 = None
    
    if figure == 1:
        subj1 = A
        obj1 = B
        subj2 = B
        obj2 = C
    elif figure == 2:
        subj1 = B
        obj1 = A
        subj2 = C
        obj2 = B
    elif figure == 3:
        subj1 = A
        obj1 = B
        subj2 = C
        obj2 = B
    elif figure == 4:
        subj1 = B
        obj1 = A
        subj2 = B
        obj2 = C
    
    p1_pos, p1_neg = get_premise_meanings(quant1, subj1, obj1)
    p2_pos, p2_neg = get_premise_meanings(quant2, subj2, obj2)

    positives = p1_pos + p2_pos
    p1_p2_neg = np.asarray(p1_neg + p2_neg, dtype="object")
    negatives = np.unique(p1_p2_neg).tolist()

    all_parts_tuples = set()
    for pos in positives:
        for p_sub in pos:
            all_parts_tuples.add(tuple(p_sub))
    for neg in negatives:
        all_parts_tuples.add(tuple(neg))

    terms_array = np.array(all_terms)
    additions = []
    for i in range(1, 2**len(all_terms)):
        mask = int_to_bools(i, len(all_terms))
        other = tuple(terms_array[mask]) 
        if other not in all_parts_tuples:
            additions.append(other)

    additions = np.array(additions, dtype="object")


    cleaned_worlds = []
    for world in positives:
        cleaned_world = [x for x in world if x not in negatives]
        cleaned_worlds.append(cleaned_world)

    final_worlds = all_combinations(cleaned_worlds)

    # expand final worlds by adding all combinations of additions
    result = []
    for world in final_worlds:
        for i in range(2**len(additions)):
            mask = int_to_bools(i, len(additions))
            to_add = additions[mask]
            world_copy = world.copy()
            for a in to_add:
                a = tuple(a)
                world_copy.add(a)
            result.append(world_copy)

    return result

def check_conclusion_in_world(conclusion, world):
    quant = conclusion[0]
    subj = None
    obj = None
    
    if conclusion.endswith("ac"):
        subj = A
        obj = C
    elif conclusion.endswith("ca"):
        subj = C
        obj = A
    
    if quant == "A":
        # if subj, then obj
        contains_subj = [x for x in world if subj in x]
        contains_obj = [obj in x for x in contains_subj]
        if not contains_obj:
            return False
        return np.all(contains_obj)
    elif quant == "E":
        # if subj, then not obj
        contains_subj = [x for x in world if subj in x]
        contains_not_obj = [obj not in x for x in contains_subj]
        if not contains_not_obj:
            return False
        return np.all(contains_not_obj)
    elif quant == "I":
        # if subj, then obj has to occur
        contains_subj = [x for x in world if subj in x]
        contains_obj = [obj in x for x in contains_subj]
        if not contains_obj:
            return False
        return np.any(contains_obj)
    elif quant == "O":
        # if subj, not obj has to occur
        contains_subj = [x for x in world if subj in x]
        contains_not_obj = [obj not in x for x in contains_subj]
        return np.any(contains_not_obj)    
     

def check_conclusion(conclusion, worlds):
    follows = True
    possible = False
    for world in worlds:
        holds = check_conclusion_in_world(conclusion, world)
        if holds:
            possible = True
        else:
            follows = False
    return (possible, follows)


def get_conclusions_for_syllog(syl):
    worlds_for_syllog = create_world_set(syl)
    
    conclusions_dict = {}
    # check conclusions
    for conclusion in ccobra.syllogistic.RESPONSES:
        if conclusion == "NVC":
            continue
        
        possible, follows = check_conclusion(conclusion, worlds_for_syllog)
        
        conclusions_dict[conclusion] = (possible, follows)
    return conclusions_dict
 

def evaluate_conclusion(conclusion, syl):
    worlds_for_syllog = create_world_set(syl)
    possible, follows = check_conclusion(conclusion, worlds_for_syllog)
    return possible, follows
    

def get_valid_responses(syl):
    concls = get_conclusions_for_syllog(syl)
    valids = [x for x, y in concls.items() if y[1]]
    if not valids:
        return ["NVC"]
    return valids

def compare_to_fol():
    for syl in ccobra.syllogistic.SYLLOGISMS:
        print("Syllogism: ", syl)
        print("    FOL:    ", ",".join(sorted(ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[syl])))
        print("    Worlds: ", ",".join(sorted(get_valid_responses(syl))))
        if ",".join(sorted(ccobra.syllogistic.SYLLOGISTIC_FOL_RESPONSES[syl])) != \
            ",".join(sorted(get_valid_responses(syl))):
            print("    !!!!!!!")
        print()

def get_possible_responses(syl):
    concl = get_conclusions_for_syllog(syl)
    possibles = [k for k, v in concl.items() if v[0] or v[1]]
    if not possibles:
        return ["NVC"]
    return possibles
