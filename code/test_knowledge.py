from knowledge_manager import generate_knowledge_assignment

# Test all 12 core configurations
configs = [
    ([0, 2, 4], "all", "nested"),
    ([0, 2, 4], "all", "disjoint"),
    ([0, 2, 4], "all", "O3"),
    ([0, 2, 4], "all", "O4"),
    ([1, 2, 3], "all", "nested"),
    ([1, 2, 3], "all", "disjoint"),
    ([1, 2, 3], "all", "O3"),
    ([1, 2, 3], "all", "O4"),
    ([2, 2, 2], "all", "nested"),
    ([2, 2, 2], "all", "disjoint"),
    ([2, 2, 2], "all", "O3"),
    ([2, 2, 2], "all", "O4"),
]

for counts, source, overlap in configs:
    assignments = generate_knowledge_assignment(counts, source, overlap, seed=42)

    # Check counts
    for i, (count, assignment) in enumerate(zip(counts, assignments)):
        assert len(assignment) == count, f"FAIL {counts}/{overlap}: Agent {i} expected {count} items, got {len(assignment)}"

    items_a = {item for item, _, _ in assignments[0]}
    items_b = {item for item, _, _ in assignments[1]}
    items_c = {item for item, _, _ in assignments[2]}

    # Check overlap patterns
    if overlap == "nested":
        assert items_a <= items_b, f"FAIL nested: A not subset of B"
        assert items_b <= items_c, f"FAIL nested: B not subset of C"
    elif overlap == "disjoint":
        assert items_a & items_b == set(), f"FAIL disjoint: A and B overlap"
        assert items_a & items_c == set(), f"FAIL disjoint: A and C overlap"
        assert items_b & items_c == set(), f"FAIL disjoint: B and C overlap"
    elif overlap == "O3":
        assert items_a & items_c == set(), f"FAIL O3: A and C should not overlap"
    elif overlap == "O4":
        assert items_b & items_c == set(), f"FAIL O4: B and C should not overlap"

    print(f"PASS: counts={counts}, source={source}, overlap={overlap}")
    for i, label in enumerate(["A", "B", "C"]):
        item_names = [item for item, _, _ in assignments[i]]
        print(f"  Agent {label}: {item_names}")
    print()

# Test source filtering
for source in ["all", "top", "bottom"]:
    assignments = generate_knowledge_assignment([2, 2, 4], "all" if source == "all" else source, "disjoint", seed=42)
    for i, assignment in enumerate(assignments):
        for item, rank, _ in assignment:
            if source == "top":
                assert rank <= 8, f"FAIL source=top: item '{item}' has rank {rank}"
            elif source == "bottom":
                assert rank >= 8, f"FAIL source=bottom: item '{item}' has rank {rank}"
    print(f"PASS: source={source} filtering correct")

print("\nAll tests passed!")
