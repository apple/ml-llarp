#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import inspect
import os.path as osp
from typing import Dict, List, Tuple

import yaml
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import (
    LogicalExpr, LogicalQuantifierType)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType, PddlEntity, SimulatorObjectType)
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaModel, LlamaTokenizer,
                          T5Model)

import llarp.config
import llarp.dataset

# Also defined in the PDDL
PLACABLE_RECEP_TYPE = "place_receptacle"


def get_allowed_actions(pddl, allowed_substrings):
    all_actions = pddl.get_possible_actions()

    def matches_any(s, allowed_acs):
        # returns if the string starts with any strings from allowed_acs
        return any(s.name.startswith(ac) for ac in allowed_acs)

    return [ac for ac in all_actions if matches_any(ac, allowed_substrings)]


def _recur_replace(expr, search, replace):
    if isinstance(expr, LogicalExpr):
        for subexpr in expr.sub_exprs:
            _recur_replace(subexpr, search, replace)
    else:
        for i, arg_val in enumerate(expr._arg_values):
            if arg_val == search:
                expr._arg_values[i] = replace


def flatten_actions(pddl: PddlDomain, obj_cats):
    new_acs = {}
    for ac_name, action in pddl.actions.items():
        found_i = -1
        # TODO: Currently this is a hack for only the pick action. This will
        # not work for other PDDL actions.

        for i, param in enumerate(action.params):
            if param.expr_type.name == SimulatorObjectType.MOVABLE_ENTITY.value:
                found_i = i
                break

        if found_i == -1:
            new_acs[ac_name] = action
            continue

        param = action.params[found_i]
        del action.params[found_i]
        for obj_cat in obj_cats:
            precond = action.precond.clone()
            assert len(precond.inputs) == 0, precond.quantifier is None

            obj_cat_type = pddl.expr_types[obj_cat]
            at_entity = PddlEntity(name="DYN_OBJ", expr_type=obj_cat_type)
            inputs = [at_entity]

            # Ignore the first expression which was about the robot position.
            precond = precond.sub_in({param: at_entity})

            postcond_pred = pddl.parse_predicate(
                f"holding({at_entity.name})", {at_entity.name: at_entity}
            )
            obj_action = PddlAction(
                f"{action.name}_{obj_cat}",
                action._params,
                pre_cond=LogicalExpr(
                    precond.expr_type,
                    precond.sub_exprs,
                    inputs,
                    LogicalQuantifierType.EXISTS,
                ),
                post_cond=[postcond_pred],
            )

            new_acs[obj_action.name] = obj_action
    pddl.set_actions(new_acs)
    return pddl


def get_obj_type(cat, pddl):
    return pddl.expr_types[SimulatorObjectType.MOVABLE_ENTITY.value]


def get_pddl(task_config, all_cats, obj_cats) -> PddlDomain:
    config_path = osp.dirname(inspect.getfile(llarp.config))
    domain_file_path = osp.join(
        config_path,
        task_config.task_spec_base_path,
        task_config.pddl_domain_def + ".yaml",
    )
    pddl = PddlDomain(
        domain_file_path,
        task_config,
    )

    # Add permanent entity types. (Object types and receptacles).
    for cat in all_cats:
        if cat in obj_cats:
            obj_type = get_obj_type(cat, pddl)
            entity_type = ExprType(cat, obj_type)
            pddl._expr_types[cat] = entity_type
        else:
            # Assume this is a receptacle in the scene. Permanently place, not per episode.
            pddl._constants[cat] = PddlEntity(cat, pddl.expr_types[PLACABLE_RECEP_TYPE])

    return flatten_actions(pddl, obj_cats)


def get_parser(llm_id):
    if "llama" in llm_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(llm_id)
        # llama has no pad token by default. As per this thread:
        # https://github.com/huggingface/transformers/issues/22312 we should
        # set pad token manually.
        tokenizer.pad_token = "[PAD]"
        return tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
        tokenizer.pad_token = "[PAD]"
        return tokenizer
