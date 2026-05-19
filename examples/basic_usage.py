"""
basic_usage.py — end-to-end demonstration of the 3D bin packing library.

Run with:
    python examples/basic_usage.py              # optimal (heuristic + GA)
    python examples/basic_usage.py sequential   # sequential / order-preserving
"""

from bin_packing import Cuboid, Packer, PackerConfig
from bin_packing.genetic import GAConfig


def main():
    print("Packing mode:")
    print("  1. Optimal  (heuristic + GA, items sorted by volume)")
    print("  2. Sequential (arrival order, stops on first failure)")
    choice = input("Choose [1/2]: ").strip()
    mode = "sequential" if choice == "2" else "optimal"
    print()
    # Container: 10 × 10 × 10 units (narrow footprint, tall — forces vertical stacking)
    container = (10, 10, 5)

    # Items to pack
    cuboids = [
        Cuboid("pallet_A",  dims=(8, 6, 5)),
        Cuboid("pallet_B",  dims=(6, 6, 4)),
        Cuboid("pallet_C",  dims=(10, 4, 3)),
        Cuboid("box_1",     dims=(4, 4, 4)),
        Cuboid("box_2",     dims=(3, 5, 6)),
        Cuboid("box_3",     dims=(5, 3, 4)),
        Cuboid("small_1",   dims=(2, 2, 3)),
        Cuboid("small_2",   dims=(3, 2, 2)),
        Cuboid("small_3",   dims=(1, 4, 2)),
        Cuboid("flat_1",    dims=(6, 5, 1)),
    ]

    total_item_volume = sum(c.volume for c in cuboids)
    container_volume = container[0] * container[1] * container[2]
    print(f"Container:         {container[0]}×{container[1]}×{container[2]} = {container_volume} units³")
    print(f"Total item volume: {total_item_volume} units³")
    print(f"Theoretical max:   {total_item_volume / container_volume:.1%}")
    print()

    packer = Packer(container, PackerConfig(use_ga=False))

    if mode == "sequential":
        # --- Sequential: strict arrival order, stop on first failure ---
        result = packer.pack_sequential(list(cuboids))
        print("=== Sequential (arrival order) ===")
        print(f"  Packed:      {len(result.placements)}/{len(cuboids)} items")
        print(f"  Utilisation: {result.utilisation:.1%}")
        print(f"  Max height:  {result.h_max}")
        if result.unpacked_ids:
            print(f"  Unplaced:    {result.unpacked_ids}")
        print()
    else:
        # --- Heuristic only ---
        h_result = packer.pack(list(cuboids))
        print("=== Heuristic only ===")
        print(f"  Packed:      {len(h_result.placements)}/{len(cuboids)} items")
        print(f"  Utilisation: {h_result.utilisation:.1%}")
        print(f"  Max height:  {h_result.h_max}")
        if h_result.unpacked_ids:
            print(f"  Unplaced:    {h_result.unpacked_ids}")
        print()

        # --- Hybrid: heuristic + GA ---
        ga_config = GAConfig(
            population_size=40,
            n_generations=100,
            plateau_patience=20,
        )
        ga_packer = Packer(container, PackerConfig(use_ga=True, ga_config=ga_config), seed=42)
        result = ga_packer.pack(list(cuboids))
        print("=== Hybrid (heuristic + GA) ===")
        print(f"  Packed:      {len(result.placements)}/{len(cuboids)} items")
        print(f"  Utilisation: {result.utilisation:.1%}")
        print(f"  Max height:  {result.h_max}")
        if result.unpacked_ids:
            print(f"  Unplaced:    {result.unpacked_ids}")
        print()

    # --- Placement details ---
    label = "sequential" if mode == "sequential" else "hybrid"
    print(f"=== Placements ({label} result) ===")
    for p in result.placements:
        x, y, z = p.position
        l, w, h = p.orientation.l, p.orientation.w, p.orientation.h
        print(
            f"  {p.item_id:<12}  pos=({x:>3},{y:>3},{z:>3})  "
            f"dims={l}×{w}×{h}"
        )

    # --- 3D visualisation ---
    try:
        from bin_packing.visualisation import animate_packing
        title = f"3D Bin Packing — {'Sequential' if mode == 'sequential' else 'Drop'} Animation"
        fig = animate_packing(result, title=title)
        fig.show()  # opens an interactive browser tab; press Play to watch boxes drop
        # fig.write_html("packing.html")  # save to file instead
    except ImportError:
        print("\nInstall plotly to see the 3D visualisation:")
        print("  pip install plotly")


if __name__ == "__main__":
    main()
