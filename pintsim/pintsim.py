import copy
import random
import tempfile
import os

from pathos.pools import ProcessPool
from pyparsing import *

import pypint

class Transition(object):

    def __init__(self, initial, final, conditions, propensity=1):
        self.initial = initial
        self.final = final
        self.conditions = conditions
        self.propensity = propensity

    def __str__(self):

        def _value_to_str(value):
            if isinstance(value, int):
                return str(value)
            else:
                return '"{}"'.format(value)

        def _name_to_str(name):
            return '"{}"'.format(name)

        s_initial_final = "{}".format(
            ", ".join(["{} {} -> {}".format(
                _name_to_str(name),
                _value_to_str(self.initial[name]),
                _value_to_str(self.final[name])) for name in self.initial]))
        if self.conditions:
            s_conditions = " when {}".format(" and ".join(["{}={}".format(
                _name_to_str(name),
                _value_to_str(self.conditions[name])) for name in self.conditions]))
        else:
            s_conditions = ""
        return "{}{}".format(s_initial_final, s_conditions)

class Automaton(object):

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name and self.values == other.values

    def __str__(self):
        def _value_to_str(value):
            if isinstance(value, int):
                return str(value)
            else:
                return '"{}"'.format(value)

        def _name_to_str(name):
            return '"{}"'.format(name)

        s = "{} [{}]".format(
            _name_to_str(self.name),
            ", ".join([_value_to_str(value) for value in self.values])
        )
        return s

class Trace(object):

    def __init__(self, first_state, last_state=None, transitions=None):
        self.first_state = first_state
        self.last_state = last_state if last_state is not None else first_state
        self.transitions = transitions if transitions is not None else []

    def add_transition(self, transition):
        self.transitions.append(transition)

    def __add__(self, other):
        trace = copy.copy(self)
        trace.transitions += other.transitions
        trace.last_state = other.last_state
        return trace

    def __len__(self):
        return len(self.transitions) + 1

    def __str__(self):
        return "FROM {} TO {} WITH {}".format(self.first_state,
                self.last_state, "; ".join([str(transition) for transition in
                    self.transitions]))

    def __getitem__(self, n):
        current_state = self.first_state
        for i in range(n - 1):
            current_state = apply_transition(self.transitions[i],
                    current_state)
        return current_state

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            if self.n == 0:
                self.current_state = self.first_state
            else:
                self.current_state = apply_transition(self.transitions[self.n -
                    1], self.current_state)
            self.n += 1
            return self.current_state
        raise StopIteration

class Model(object):

    def __init__(self, automata=None, transitions=None):
        self.automata = automata
        self.transitions = transitions

    def to_pypint(self, initial_state=None):
        def _value_to_str(value):
            if isinstance(value, int):
                return str(value)
            else:
                return '"{}"'.format(value)

        def _name_to_str(name):
            return '"{}"'.format(name)

        file_name = tempfile.mkstemp(suffix=".an")[1]
        with open(file_name, "w") as f:
            for automaton in self.automata:
                f.write("{}\n".format(automaton))
            for transition in self.transitions:
                f.write("{}\n".format(transition))
            if initial_state is not None:
                f.write("initial_state {}".format(", ".join(["{}={}".format(_name_to_str(name), _value_to_str(value)) for name, value in initial_state.items()])))
        model = pypint.load(file_name)
        return model

class Goal(object):

    def __init__(self, goal=None):
        self.subgoals = []
        if goal is not None:
            if isinstance(goal, dict):
                self.subgoals.append([goal])
            elif isinstance(goal, list):
                for subgoal in goal:
                    if isinstance(subgoal, dict):
                        self.subgoals.append([subgoal])
                    elif isinstance(subgoal, Goal):
                        self.subgoals += subgoal.subgoals

    def __or__(self, other):
        if isinstance(other, dict):
            other = Goal(other)
        goal = Goal()
        goal.subgoals = [self.subgoals[0] + other.subgoals[0]]
        return goal

def load(*args, **kwargs):
    pypint_an = pypint.load(*args, **kwargs)
    model = pypint_to_model(pypint_an)
    return model

def pypint_to_model(pypint_an):
    model = Model()
    model.automata = [Automaton(name, pypint_an.local_states[name]) for name in
            pypint_an.automata]
    model.transitions = [Transition({local_transition.a: local_transition.i},
        {local_transition.a: local_transition.j}, {key: value for key, value in
            local_transition.conds.items()}) for local_transition in pypint_an.local_transitions]
    return model

def complete_state(state, model):
    for automaton in model.automata:
        if automaton.name not in state:
            state[automaton.name] = 0
    return state

def is_substate(substate, state):
    return all(item in state.items() for item in substate.items())

def is_applicable(transition, state):
    return is_substate(transition.initial, state) and is_substate(transition.conditions, state)

def get_applicable(model, state):
    return [transition for transition in model.transitions if is_applicable(transition, state)]

def apply_transition(transition, state):
    new_state = state.copy()
    for name, value in transition.final.items():
        new_state[name] = value
    return new_state

def choose(applicable):
    return random.choices(applicable, [transition.propensity for transition in applicable])[0]

def check_trace(trace):
    current_state = trace.first_state
    for transition in trace.transitions:
        if not is_applicable(transition, current_state):
            return False
        current_state = apply_transition(transition, current_state)
    if not current_state == trace.last_state:
        return False
    return True

def check_trace_pint(an, trace):
    pypint_an = an.to_pypint()
    for i, state in enumerate(trace):
        if i == 0:
            initial_state = state
        else:
            reached = pypint_an.having(initial_state).reachability(state, fallback="mole")
            if reached is False:
                return False
            elif reached == pypint.Inconc:
                return pypint.Inconc
            initial_state = state
    return True

def trace_from_string(s):
    def a_pquoted_string(toks):
        return toks[0]

    def a_pname(toks):
        return toks[0]

    def a_pstring_value(toks):
        return toks[0]

    def a_pint_value(toks):
        return int(toks[0])

    def a_pvalue(toks):
        return toks[0]

    def a_pstate_item(toks):
        # print((toks.name, toks.value))
        return (toks.name, toks.value)

    def a_pstate(toks):
        return dict(list(toks.state_items))

    def a_pcondition_item(toks):
        return (toks.name, toks.value)

    def a_pcondition(toks):
        return dict(list(toks))

    def a_ptransition(toks):
        transition = Transition({toks.name: toks.initial},
                {toks.name: toks.final}, toks.condition if toks.condition else
                {})
        return transition

    def a_ptrace(toks):
        trace = Trace(toks.first_state, toks.last_state, list(toks.transitions))
        return trace

    pquoted_string = QuotedString(quoteChar='"', unquoteResults=True) | QuotedString(quoteChar="'", unquoteResults=True)
    pquoted_string.setParseAction(a_pquoted_string)

    pname = pquoted_string
    pname.setParseAction(a_pname)

    pstring_value = pquoted_string
    pstring_value.setParseAction(a_pstring_value)

    pint_value = Word(nums)
    pint_value.setParseAction(a_pint_value)

    pvalue = pstring_value | pint_value
    pvalue.setParseAction(a_pvalue)

    pstate_item = pname("name") + Literal(": ") + pvalue("value")
    pstate_item.setParseAction(a_pstate_item)

    pstate = Char("{") + delimitedList(pstate_item, delim=", ")("state_items") + Char("}")
    pstate.setParseAction(a_pstate)

    pcondition_item = pname("name") + Char("=") + pvalue("value")
    pcondition_item.setParseAction(a_pcondition_item)

    pcondition = delimitedList(pcondition_item, delim=" and ")
    pcondition.setParseAction(a_pcondition)

    ptransition = pname("name") + Char(" ") + pvalue("initial") + Literal(" -> ") + pvalue("final") + Optional(Literal(" when ") + pcondition("condition"))
    ptransition.setParseAction(a_ptransition)

    ptrace = Literal("FROM ") + pstate("first_state") + Literal(" TO ") + pstate("last_state") + Literal(" WITH ") + delimitedList(ptransition, delim="; ")("transitions")
    ptrace.setParseAction(a_ptrace)

    ptrace.leaveWhitespace()

    trace = ptrace.parseString(s, parseAll=True)[0]
    return trace

def reachability(model, from_state, goal, max_length=2000, on_start=None, on_reach=None, max_repeat=10000, n_workers=1):
    if isinstance(model, pypint.Model):
        model = pypint_to_model(model)
    if isinstance(goal, list) or isinstance(goal, dict):
        goal = Goal(goal)
    if isinstance(from_state, list):
        if from_state:
            if isinstance(from_state[0], str):
                from_state = dict([(e, 1) for e in from_state])
            elif isinstance(from_state[0], tuple):
                from_state = dict(from_state)
    from_state = complete_state(from_state, model)
    trace = Trace(from_state)
    if on_start is not None:
        next_subgoal = goal.subgoals[0]
        on_start(model, trace, next_subgoal)
    if n_workers == 1:
        for n_repeat in range(max_repeat):
            reached, trace = _reach(copy.copy(model), from_state, goal, max_length, on_start, on_reach)
            if reached is True:
                return reached, trace
    else:
        pool = ProcessPool(n_workers)
        processes = set([])
        n_repeat = 0
        while n_repeat < max_repeat and n_repeat < n_workers:
            processes.add(pool.apipe(_reach, copy.copy(model), from_state, goal, max_length, on_start, on_reach))
            n_repeat += 1
        reached = pypint.Inconc
        while reached is not True and n_repeat < max_repeat:
            for process in processes:
                if process.ready():
                    reached, trace = process.get()
                    processes.remove(process)
                    if reached is True:
                        return reached, trace
                    else:
                        processes.add(pool.apipe(_reach, copy.copy(model), from_state, goal, max_length, on_start, on_reach))
                        n_repeat += 1
                        break
    return reached, trace

def _reach(model, from_state, goal, max_length, on_start, on_reach):
    trace = Trace(from_state)
    # if on_start is not None:
    #     next_subgoal = goal.subgoals[0]
    #     on_start(model, trace, next_subgoal)
    reached = True
    for n_subgoal, subgoal in enumerate(goal.subgoals):
        subreached, subtrace = _reach_subgoal(model, trace.last_state, subgoal, max_length)
        trace += subtrace
        if subreached is True:
            if n_subgoal < len(goal.subgoals) - 1:
                next_subgoal = goal.subgoals[n_subgoal + 1]
            else:
                next_subgoal = None
            if on_reach is not None:
                on_reach(model, trace, subgoal, next_subgoal)
        else:
            reached = pypint.Inconc
            break
    return reached, trace

def _reach_subgoal(model, from_state, subgoal, max_length=5000):
    trace = Trace(from_state)
    current_state = from_state
    for disj_subgoal in subgoal:
        if is_substate(disj_subgoal, current_state):
            reached = True
            return reached, trace
    reached = pypint.Inconc
    n_transitions = 0
    while len(trace) < max_length:
        applicable = get_applicable(model, current_state)
        if len(applicable) > 0:
            transition = choose(applicable)
            current_state = apply_transition(transition, current_state)
            trace.add_transition(transition)
            for disj_subgoal in subgoal:
                if is_substate(disj_subgoal, current_state):
                    trace.last_state = current_state
                    reached = True
                    return reached, trace
        else:
            break
    trace.last_state = current_state
    return reached, trace
