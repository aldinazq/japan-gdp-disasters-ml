"""I inspect which years in my master dataset have non-zero disaster features."""

from src.features import build_master_table


def main() -> None:
    df = build_master_table()
    print("Years with at least one disaster:\n")
    print(df[df["n_events"] > 0].head(30))


if __name__ == "__main__":
    main()
