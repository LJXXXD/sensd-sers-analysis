import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import re


class RamanMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman TXT to Excel Merger")
        self.file_paths = []
        self.cfu_entries = []

        # File selection
        self.select_button = tk.Button(
            root, text="📂 Select TXT Files", command=self.select_files, width=25
        )
        self.select_button.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # File list + CFU inputs area
        self.files_frame = tk.Frame(root)
        self.files_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Raman shift range
        tk.Label(root, text="Min Raman Shift (optional):").grid(
            row=2, column=0, padx=10, sticky="e"
        )
        self.min_shift_entry = tk.Entry(root)
        self.min_shift_entry.grid(row=2, column=1, padx=10, sticky="ew")

        tk.Label(root, text="Max Raman Shift (optional):").grid(
            row=3, column=0, padx=10, sticky="e"
        )
        self.max_shift_entry = tk.Entry(root)
        self.max_shift_entry.grid(row=3, column=1, padx=10, sticky="ew")

        # Export button
        self.run_button = tk.Button(
            root, text="🚀 Convert and Export", command=self.run_merge, width=25
        )
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20)

        # Enable dynamic resizing
        for i in range(5):
            root.grid_rowconfigure(i, weight=1)
        root.grid_columnconfigure(1, weight=1)

    def select_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])

        # Sort files numerically by first number in filename
        def extract_cfu_number(filename):
            match = re.search(r"(\d+)", filename)
            return int(match.group(1)) if match else float("inf")

        self.file_paths = sorted(file_paths, key=lambda x: extract_cfu_number(os.path.basename(x)))

        # Clear previous UI
        for widget in self.files_frame.winfo_children():
            widget.destroy()
        self.cfu_entries = []

        # Headers
        tk.Label(self.files_frame, text="Selected File").grid(row=0, column=0, padx=5, sticky="w")
        tk.Label(self.files_frame, text="CFU/mL Value").grid(row=0, column=1, padx=5)

        # Add each file with entry box
        for i, path in enumerate(self.file_paths):
            tk.Label(self.files_frame, text=os.path.basename(path), anchor="w").grid(
                row=i + 1, column=0, sticky="nsew"
            )
            entry = tk.Entry(self.files_frame)
            entry.grid(row=i + 1, column=1, sticky="nsew")
            self.cfu_entries.append(entry)

        # Resize behavior for file input area
        self.files_frame.grid_columnconfigure(0, weight=3)
        self.files_frame.grid_columnconfigure(1, weight=1)

    def run_merge(self):
        if not self.file_paths:
            messagebox.showerror("Error", "No files selected.")
            return

        # Gather CFU values
        cfu_values = []
        for i, entry in enumerate(self.cfu_entries):
            val = entry.get().strip()
            if not val.isdigit():
                messagebox.showerror(
                    "Error", f"Invalid CFU value for file {os.path.basename(self.file_paths[i])}"
                )
                return
            cfu_values.append(val)

        # Raman shift range
        try:
            min_shift = (
                float(self.min_shift_entry.get()) if self.min_shift_entry.get().strip() else None
            )
            max_shift = (
                float(self.max_shift_entry.get()) if self.max_shift_entry.get().strip() else None
            )
        except ValueError:
            messagebox.showerror("Error", "Raman shift values must be numeric.")
            return

        # Load and check files
        dataframes = []
        common_shift = None

        for i, path in enumerate(self.file_paths):
            df = pd.read_csv(
                path, sep="\t", comment=">", header=None, names=["RamanShift", "Value"]
            )
            df["RamanShift"] = pd.to_numeric(df["RamanShift"], errors="coerce")
            df = df.dropna()

            if common_shift is None:
                common_shift = df["RamanShift"]
            else:
                if not np.allclose(common_shift, df["RamanShift"], atol=1e-4):
                    messagebox.showerror(
                        "Error", f"Raman shift mismatch in file: {os.path.basename(path)}"
                    )
                    return
            dataframes.append(df)

        # Determine truncation range
        min_final = min_shift if min_shift is not None else common_shift.min()
        max_final = max_shift if max_shift is not None else common_shift.max()
        if min_final >= max_final:
            messagebox.showerror("Error", "Min Raman Shift must be less than Max.")
            return

        # Merge
        merged_df = pd.DataFrame()
        merged_df["Ramen Shift"] = common_shift
        merged_df = merged_df[
            (merged_df["Ramen Shift"] >= min_final) & (merged_df["Ramen Shift"] <= max_final)
        ]
        merged_df.reset_index(drop=True, inplace=True)

        for i, df in enumerate(dataframes):
            df = df[(df["RamanShift"] >= min_final) & (df["RamanShift"] <= max_final)]
            df.reset_index(drop=True, inplace=True)
            col = int(cfu_values[i])  # Use integer as column name
            merged_df[col] = df["Value"]

        # Export to Excel
        output_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")]
        )
        if output_path:
            merged_df.to_excel(output_path, index=False)
            messagebox.showinfo("Success", f"Merged Excel file saved:\n{output_path}")


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RamanMergerApp(root)
    root.mainloop()
