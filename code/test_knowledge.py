from knowledge_manager import generate_knowledge_assignment

# Phase 8 condition matrix: 2 configs × 5 overlaps. Source variants removed.
configs = [
    ([0, 2, 4], "nested"),
    ([0, 2, 4], "disjoint"),
    ([0, 2, 4], "O3"),
    ([0, 2, 4], "O4"),
    ([0, 2, 4], "O5"),
    ([2, 2, 2], "nested"),
    ([2, 2, 2], "disjoint"),
    ([2, 2, 2], "O3"),
    ([2, 2, 2], "O4"),
    ([2, 2, 2], "O5"),
]

for counts, overlap in configs:
    assignments = generate_knowledge_assignment(counts, overlap, seed=42)

    # Check counts
    for i, (count, assignment) in enumerate(zip(counts, assignments)):
        assert len(assignment) == count, f"FAIL {counts}/{overlap}: Agent {i} expected {count} items, got {len(assignment)}"

    items_a = {item for item, _, _ in assignments[0]}
    items_b = {item for item, _, _ in assignments[1]}
    items_c = {item for item, _, _ in assignments[2]}

    # Check overlap patterns
    if overlap == "nested":
        assert items_a <= items_b, f"FAIL nested {counts}: A not subset of B"
        assert items_b <= items_c, f"FAIL nested {counts}: B not subset of C"
    elif overlap == "disjoint":
        assert items_a & items_b == set(), f"FAIL disjoint {counts}: A and B overlap"
        assert items_a & items_c == set(), f"FAIL disjoint {counts}: A and C overlap"
        assert items_b & items_c == set(), f"FAIL disjoint {counts}: B and C overlap"
    elif overlap == "O3":
        # B ⊆ C; A disjoint from both
        assert items_b <= items_c, f"FAIL O3 {counts}: B not subset of C"
        assert items_a & items_b == set(), f"FAIL O3 {counts}: A and B overlap"
        assert items_a & items_c == set(), f"FAIL O3 {counts}: A and C overlap"
    elif overlap == "O4":
        # A ⊆ C; B disjoint from both
        assert items_a <= items_c, f"FAIL O4 {counts}: A not subset of C"
        assert items_a & items_b == set(), f"FAIL O4 {counts}: A and B overlap"
        assert items_b & items_c == set(), f"FAIL O4 {counts}: B and C overlap"
    elif overlap == "O5":
        # smaller of A/B is a subset of the larger; C disjoint from both
        if len(items_a) <= len(items_b):
            assert items_a <= items_b, f"FAIL O5 {counts}: A not subset of B"
        else:
            assert items_b <= items_a, f"FAIL O5 {counts}: B not subset of A"
        assert items_a & items_c == set(), f"FAIL O5 {counts}: A and C overlap"
        assert items_b & items_c == set(), f"FAIL O5 {counts}: B and C overlap"

    print(f"PASS: counts={counts}, overlap={overlap}")
    for i, label in enumerate(["A", "B", "C"]):
        item_names = [item for item, _, _ in assignments[i]]
        print(f"  Agent {label}: {item_names}")
    print()

# Sanity check: O5 with [2,2,2] should give A == B (set equality)
a, b, c = generate_knowledge_assignment([2, 2, 2], "O5", seed=42)
items_a = {item for item, _, _ in a}
items_b = {item for item, _, _ in b}
assert items_a == items_b, f"FAIL O5 [2,2,2]: A and B should be identical"
print("PASS: O5 [2,2,2] yields A == B")

print("\nAll tests passed!")
