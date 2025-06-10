import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def read_csv_with_encoding(file_path):
    """
    Try to read CSV file with different encodings
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252", "gbk", "gb2312"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(
                f"Successfully read file using {encoding} encoding: {os.path.basename(file_path)}"
            )
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {e}")
            continue

    raise Exception(f"Unable to read file with any encoding: {file_path}")


def analyze_grade_distribution():
    """
    Analyze Grade distribution in all CSV files in the dataset directory
    """
    # Set font
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Get all CSV file paths
    csv_files = glob.glob("dataset/*.csv")
    print(f"Found {len(csv_files)} CSV files")

    # Store all Grade data
    all_grades = []
    file_stats = []

    # Read each file and extract Grade data
    for file_path in csv_files:
        try:
            # Try to read CSV file with different encodings
            df = read_csv_with_encoding(file_path)

            # Find Grade column (may be named 'Grade' or ' Grade')
            grade_col = None
            for col in df.columns:
                if col.strip().lower() == "grade":
                    grade_col = col
                    break

            if grade_col is not None:
                grades = df[grade_col].dropna()

                # Data cleaning: filter valid Grade values (0-5)
                valid_grades = grades[(grades >= 0) & (grades <= 5)]
                invalid_grades = grades[(grades < 0) | (grades > 5)]

                if len(invalid_grades) > 0:
                    print(
                        f"  Warning: Found {len(invalid_grades)} invalid Grade values: {sorted(invalid_grades.unique())}"
                    )

                all_grades.extend(valid_grades.tolist())

                # Calculate Grade distribution for this file
                grade_counts = valid_grades.value_counts().sort_index()
                file_name = os.path.basename(file_path)

                file_stats.append(
                    {
                        "file": file_name,
                        "total_records": len(df),
                        "valid_grades": len(valid_grades),
                        "invalid_grades": len(invalid_grades),
                        "grade_distribution": grade_counts.to_dict(),
                        "mean_grade": valid_grades.mean(),
                        "std_grade": valid_grades.std(),
                    }
                )

                print(f"File: {file_name}")
                print(f"  Total records: {len(df)}")
                print(f"  Valid Grade count: {len(valid_grades)}")
                if len(invalid_grades) > 0:
                    print(f"  Invalid Grade count: {len(invalid_grades)}")
                print(f"  Grade distribution: {dict(grade_counts)}")
                print(f"  Mean Grade: {valid_grades.mean():.2f}")
                print(f"  Grade std dev: {valid_grades.std():.2f}")
                print("-" * 50)
            else:
                print(
                    f"File {os.path.basename(file_path)} does not contain 'Grade' column"
                )
                print(f"  Available columns: {df.columns.tolist()}")

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Check if we have any data
    if not all_grades:
        print("Error: No Grade data found")
        return None

    # Convert to pandas Series for overall analysis
    all_grades_series = pd.Series(all_grades)

    print("\n=== Overall Grade Distribution Analysis ===")
    print(f"Total Grade records: {len(all_grades_series)}")
    print(f"Grade range: {all_grades_series.min()} - {all_grades_series.max()}")
    print(f"Mean Grade: {all_grades_series.mean():.2f}")
    print(f"Median Grade: {all_grades_series.median():.2f}")
    print(f"Grade std dev: {all_grades_series.std():.2f}")

    # Grade distribution statistics
    grade_distribution = all_grades_series.value_counts().sort_index()
    print("\nGrade distribution statistics:")
    for grade, count in grade_distribution.items():
        percentage = count / len(all_grades_series) * 100
        print(f"Grade {grade}: {count} times ({percentage:.2f}%)")

    # Create visualization charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Grade Distribution Analysis", fontsize=16, fontweight="bold")

    # 1. Grade distribution bar chart
    grade_distribution.plot(kind="bar", ax=axes[0, 0], color="skyblue", alpha=0.7)
    axes[0, 0].set_title("Grade Distribution Bar Chart")
    axes[0, 0].set_xlabel("Grade")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Grade distribution pie chart
    grade_distribution.plot(kind="pie", ax=axes[0, 1], autopct="%1.1f%%")
    axes[0, 1].set_title("Grade Distribution Pie Chart")
    axes[0, 1].set_ylabel("")

    # 3. Grade distribution histogram
    axes[1, 0].hist(
        all_grades_series,
        bins=np.arange(all_grades_series.min() - 0.5, all_grades_series.max() + 1.5, 1),
        color="lightgreen",
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 0].set_title("Grade Distribution Histogram")
    axes[1, 0].set_xlabel("Grade")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Box plot
    axes[1, 1].boxplot(
        all_grades_series,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", alpha=0.7),
    )
    axes[1, 1].set_title("Grade Distribution Box Plot")
    axes[1, 1].set_ylabel("Grade")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/grade_distribution_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    # If there are multiple files, create Grade distribution comparison chart by file
    if len(file_stats) > 1:
        fig, ax = plt.subplots(figsize=(15, 8))

        # Prepare data for heatmap
        all_unique_grades = sorted(all_grades_series.unique())

        # Create matrix data
        matrix_data = []
        file_labels = []
        for stat in file_stats:
            row = []
            for grade in all_unique_grades:
                count = stat["grade_distribution"].get(grade, 0)
                # Calculate percentage
                percentage = (
                    count / stat["valid_grades"] * 100
                    if stat["valid_grades"] > 0
                    else 0
                )
                row.append(percentage)
            matrix_data.append(row)
            file_labels.append(
                stat["file"].replace("SM16-v-SM17_", "").replace(".csv", "")
            )

        # Create heatmap (using matplotlib's imshow)
        im = ax.imshow(matrix_data, cmap="YlOrRd", aspect="auto")

        # Set ticks and labels
        ax.set_xticks(range(len(all_unique_grades)))
        ax.set_xticklabels([f"Grade {g}" for g in all_unique_grades])
        ax.set_yticks(range(len(file_labels)))
        ax.set_yticklabels(file_labels)

        # Add value annotations
        for i in range(len(file_labels)):
            for j in range(len(all_unique_grades)):
                text = ax.text(
                    j,
                    i,
                    f"{matrix_data[i][j]:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        ax.set_title("Grade Distribution by File (Percentage)")
        ax.set_xlabel("Grade")
        ax.set_ylabel("File")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Percentage")

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            "plots/grade_distribution_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    # Return analysis results
    return {
        "total_records": len(all_grades_series),
        "grade_distribution": grade_distribution.to_dict(),
        "statistics": {
            "mean": all_grades_series.mean(),
            "median": all_grades_series.median(),
            "std": all_grades_series.std(),
            "min": all_grades_series.min(),
            "max": all_grades_series.max(),
        },
        "file_stats": file_stats,
    }


if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    results = analyze_grade_distribution()
    if results:
        print(
            "\nAnalysis completed! Charts saved as 'plots/grade_distribution_analysis.png'"
        )
        if len(results["file_stats"]) > 1:
            print("Heatmap saved as 'plots/grade_distribution_heatmap.png'")
    else:
        print("Analysis failed: No valid Grade data found")
