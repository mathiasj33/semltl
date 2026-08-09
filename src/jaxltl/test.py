from pathlib import Path
from jaxltl.ltl.hoa import HOAParser

hoa = Path(
    "../fishsemml/build/test-output/embedding.hoa"
).read_text()

dba = HOAParser(
    formula="∃(F(a)) ∧ ∀(G(b))",
    hoa_text=hoa,
    propositions=("a", "b", "c"),
).parse_hoa()

print(dba.num_states)
print(dba.initial_state)
print(dba.state_to_info[0]["formula_embedding"])

dba.compute_sccs()
print(dba.is_finite_specification())

for state in dba.states:
    embedding = dba.state_to_info[state]["formula_embedding"]
    print(state, len(embedding))