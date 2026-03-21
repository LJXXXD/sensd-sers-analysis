"""
Chicken SERS Sensor Testing Protocol Analysis
============================================

This script implements the 5-experiment protocol for chicken Salmonella detection
using SERS sensors. Each experiment focuses on specific aspects of sensor performance
and data analysis with defined metrics and statistical validation.

University of Missouri Team - Chicken Solution Focus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
from datetime import datetime

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(".."))
from src.data.sers_io import load_dataset_with_serotype


class ChickenProtocolAnalyzer:
    """
    Main class for conducting the 5-experiment chicken SERS protocol analysis
    """

    def __init__(self, data_folder, output_dir="protocol_results"):
        """
        Initialize the analyzer with data folder and output directory

        Args:
            data_folder (str): Path to SERS data folder
            output_dir (str): Directory to save results
        """
        self.data_folder = data_folder
        self.output_dir = output_dir
        self.results = {}
        self.metrics_summary = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data storage
        self.data_entries = []
        self.filtered_data = {}

    def load_chicken_data(self, signals_folders=None, serotypes=None):
        """
        Load chicken-specific SERS data

        Args:
            signals_folders (list): List of signal folders to load
            serotypes (list): List of serotypes to focus on (default: ST, SE)
        """
        if signals_folders is None:
            signals_folders = ["December Signals", "SERS Signals"]

        if serotypes is None:
            serotypes = ["ST", "SE"]  # Salmonella Typhimurium and Enterica

        print("🔄 Loading chicken SERS data...")
        print(f"📁 Data folder: {self.data_folder}")
        print(f"📂 Signal folders: {signals_folders}")
        print(f"🦠 Serotypes: {serotypes}")

        try:
            self.data_entries = load_dataset_with_serotype(
                self.data_folder, signals_folders, serotypes
            )
            print(f"✅ Successfully loaded {len(self.data_entries)} data entries")

            # Filter for chicken-relevant data (you may need to adjust this based on your data)
            self.filtered_data = self._filter_chicken_data()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def _filter_chicken_data(self):
        """
        Filter data for chicken-specific analysis
        This method can be customized based on your specific data structure
        """
        filtered = {
            "all_data": self.data_entries,
            "st_data": [entry for entry in self.data_entries if entry["serotype"] == "ST"],
            "se_data": [entry for entry in self.data_entries if entry["serotype"] == "SE"],
            "high_conc": [
                entry
                for entry in self.data_entries
                if any(conc >= 100 for conc in entry["concentrations"])
            ],
            "low_conc": [
                entry
                for entry in self.data_entries
                if any(conc < 100 for conc in entry["concentrations"])
            ],
        }

        print("📊 Data filtering results:")
        for key, data in filtered.items():
            print(f"   {key}: {len(data)} entries")

        return filtered

    def experiment_1_basic_analysis(self):
        """
        Experiment 1: Basic SERS Signal Analysis
        ========================================

        Objectives:
        - Analyze basic signal characteristics
        - Calculate signal-to-noise ratios
        - Assess signal reproducibility
        - Generate baseline metrics
        """
        print("\n" + "=" * 60)
        print("🧪 EXPERIMENT 1: Basic SERS Signal Analysis")
        print("=" * 60)

        results = {
            "experiment_name": "Basic SERS Signal Analysis",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }

        # 1.1 Signal Quality Assessment
        print("\n📈 1.1 Signal Quality Assessment")
        signal_quality = self._assess_signal_quality()
        results["metrics"]["signal_quality"] = signal_quality

        # 1.2 Signal-to-Noise Ratio Analysis
        print("\n📊 1.2 Signal-to-Noise Ratio Analysis")
        snr_analysis = self._calculate_snr_metrics()
        results["metrics"]["snr_analysis"] = snr_analysis

        # 1.3 Reproducibility Assessment
        print("\n🔄 1.3 Reproducibility Assessment")
        reproducibility = self._assess_reproducibility()
        results["metrics"]["reproducibility"] = reproducibility

        # 1.4 Generate visualizations
        self._plot_experiment_1_results(signal_quality, snr_analysis, reproducibility)

        self.results["experiment_1"] = results
        print("✅ Experiment 1 completed successfully")

        return results

    def experiment_2_concentration_analysis(self):
        """
        Experiment 2: Concentration-Dependent Analysis
        ==============================================

        Objectives:
        - Analyze concentration-response relationships
        - Calculate detection limits
        - Assess linearity and sensitivity
        - Generate calibration curves
        """
        print("\n" + "=" * 60)
        print("🧪 EXPERIMENT 2: Concentration-Dependent Analysis")
        print("=" * 60)

        results = {
            "experiment_name": "Concentration-Dependent Analysis",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }

        # 2.1 Concentration-Response Analysis
        print("\n📈 2.1 Concentration-Response Analysis")
        conc_response = self._analyze_concentration_response()
        results["metrics"]["concentration_response"] = conc_response

        # 2.2 Detection Limit Calculation
        print("\n🎯 2.2 Detection Limit Calculation")
        detection_limits = self._calculate_detection_limits()
        results["metrics"]["detection_limits"] = detection_limits

        # 2.3 Linearity Assessment
        print("\n📏 2.3 Linearity Assessment")
        linearity = self._assess_linearity()
        results["metrics"]["linearity"] = linearity

        # 2.4 Sensitivity Analysis
        print("\n⚡ 2.4 Sensitivity Analysis")
        sensitivity = self._analyze_sensitivity()
        results["metrics"]["sensitivity"] = sensitivity

        # 2.5 Generate visualizations
        self._plot_experiment_2_results(conc_response, detection_limits, linearity, sensitivity)

        self.results["experiment_2"] = results
        print("✅ Experiment 2 completed successfully")

        return results

    def _assess_signal_quality(self):
        """Assess basic signal quality metrics"""
        quality_metrics = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            # Calculate signal statistics
            all_signals = np.concatenate([entry["signals"].flatten() for entry in serotype_data])

            quality_metrics[serotype] = {
                "mean_intensity": np.mean(all_signals),
                "std_intensity": np.std(all_signals),
                "cv_percent": (np.std(all_signals) / np.mean(all_signals)) * 100,
                "max_intensity": np.max(all_signals),
                "min_intensity": np.min(all_signals),
                "signal_range": np.max(all_signals) - np.min(all_signals),
                "n_samples": len(serotype_data),
            }

        return quality_metrics

    def _calculate_snr_metrics(self):
        """Calculate signal-to-noise ratio metrics"""
        snr_metrics = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            snr_values = []
            for entry in serotype_data:
                for i, conc in enumerate(entry["concentrations"]):
                    signal = entry["signals"][:, i]
                    # Calculate SNR as mean signal / std of baseline
                    baseline_std = np.std(signal[:10])  # First 10 points as baseline
                    signal_mean = np.mean(signal)
                    snr = signal_mean / baseline_std if baseline_std > 0 else 0
                    snr_values.append(snr)

            snr_metrics[serotype] = {
                "mean_snr": np.mean(snr_values),
                "std_snr": np.std(snr_values),
                "min_snr": np.min(snr_values),
                "max_snr": np.max(snr_values),
                "n_measurements": len(snr_values),
            }

        return snr_metrics

    def _assess_reproducibility(self):
        """Assess signal reproducibility across sensors and tests"""
        reproducibility_metrics = {}

        # Group by sensor and test
        sensor_groups = {}
        for entry in self.data_entries:
            key = f"S{entry['sensor_id']}_T{entry['test_id']}"
            if key not in sensor_groups:
                sensor_groups[key] = []
            sensor_groups[key].append(entry)

        # Calculate reproducibility metrics
        for serotype in ["ST", "SE"]:
            serotype_repro = []
            for group_key, group_data in sensor_groups.items():
                serotype_group = [entry for entry in group_data if entry["serotype"] == serotype]
                if len(serotype_group) > 1:
                    # Calculate coefficient of variation for this group
                    signals = [entry["signals"].flatten() for entry in serotype_group]
                    if signals:
                        mean_signal = np.mean([np.mean(sig) for sig in signals])
                        std_signal = np.std([np.mean(sig) for sig in signals])
                        cv = (std_signal / mean_signal) * 100 if mean_signal > 0 else 0
                        serotype_repro.append(cv)

            reproducibility_metrics[serotype] = {
                "mean_cv_percent": np.mean(serotype_repro) if serotype_repro else 0,
                "std_cv_percent": np.std(serotype_repro) if serotype_repro else 0,
                "n_groups": len(serotype_repro),
            }

        return reproducibility_metrics

    def _analyze_concentration_response(self):
        """Analyze concentration-response relationships"""
        conc_response = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            concentrations = []
            intensities = []

            for entry in serotype_data:
                for i, conc in enumerate(entry["concentrations"]):
                    signal = entry["signals"][:, i]
                    max_intensity = np.max(signal)
                    concentrations.append(conc)
                    intensities.append(max_intensity)

            # Calculate correlation
            if len(concentrations) > 1:
                correlation, p_value = stats.pearsonr(np.log10(concentrations), intensities)

                conc_response[serotype] = {
                    "n_points": len(concentrations),
                    "concentration_range": [min(concentrations), max(concentrations)],
                    "intensity_range": [min(intensities), max(intensities)],
                    "correlation_coefficient": correlation,
                    "p_value": p_value,
                    "is_significant": p_value < 0.05,
                }

        return conc_response

    def _calculate_detection_limits(self):
        """Calculate detection limits (LOD and LOQ)"""
        detection_limits = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            # Get blank/control measurements (assuming lowest concentration as blank)
            blank_signals = []
            sample_signals = []
            sample_concentrations = []

            for entry in serotype_data:
                for i, conc in enumerate(entry["concentrations"]):
                    signal = entry["signals"][:, i]
                    max_intensity = np.max(signal)

                    if conc <= 1:  # Assume concentrations <= 1 are blanks
                        blank_signals.append(max_intensity)
                    else:
                        sample_signals.append(max_intensity)
                        sample_concentrations.append(conc)

            if blank_signals and sample_signals:
                blank_mean = np.mean(blank_signals)
                blank_std = np.std(blank_signals)

                # LOD = 3 * blank_std
                # LOQ = 10 * blank_std
                lod = 3 * blank_std
                loq = 10 * blank_std

                detection_limits[serotype] = {
                    "blank_mean": blank_mean,
                    "blank_std": blank_std,
                    "lod": lod,
                    "loq": loq,
                    "n_blanks": len(blank_signals),
                    "n_samples": len(sample_signals),
                }

        return detection_limits

    def _assess_linearity(self):
        """Assess linearity of concentration-response relationship"""
        linearity_metrics = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            concentrations = []
            intensities = []

            for entry in serotype_data:
                for i, conc in enumerate(entry["concentrations"]):
                    signal = entry["signals"][:, i]
                    max_intensity = np.max(signal)
                    concentrations.append(conc)
                    intensities.append(max_intensity)

            if len(concentrations) > 2:
                # Fit linear regression
                log_conc = np.log10(concentrations)
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_conc, intensities
                )

                # Calculate R²
                r_squared = r_value**2

                linearity_metrics[serotype] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "p_value": p_value,
                    "std_error": std_err,
                    "is_linear": r_squared > 0.9 and p_value < 0.05,
                }

        return linearity_metrics

    def _analyze_sensitivity(self):
        """Analyze sensor sensitivity"""
        sensitivity_metrics = {}

        for serotype in ["ST", "SE"]:
            serotype_data = [entry for entry in self.data_entries if entry["serotype"] == serotype]

            if not serotype_data:
                continue

            # Calculate sensitivity as slope of concentration-response curve
            concentrations = []
            intensities = []

            for entry in serotype_data:
                for i, conc in enumerate(entry["concentrations"]):
                    signal = entry["signals"][:, i]
                    max_intensity = np.max(signal)
                    concentrations.append(conc)
                    intensities.append(max_intensity)

            if len(concentrations) > 1:
                log_conc = np.log10(concentrations)
                slope, _, r_value, p_value, _ = stats.linregress(log_conc, intensities)

                sensitivity_metrics[serotype] = {
                    "sensitivity": slope,
                    "correlation": r_value,
                    "p_value": p_value,
                    "n_points": len(concentrations),
                }

        return sensitivity_metrics

    def _plot_experiment_1_results(self, signal_quality, snr_analysis, reproducibility):
        """Generate visualizations for Experiment 1"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Experiment 1: Basic SERS Signal Analysis", fontsize=16, fontweight="bold")

        # 1.1 Signal Quality Comparison
        ax1 = axes[0, 0]
        serotypes = list(signal_quality.keys())
        mean_intensities = [signal_quality[s]["mean_intensity"] for s in serotypes]
        std_intensities = [signal_quality[s]["std_intensity"] for s in serotypes]

        ax1.bar(serotypes, mean_intensities, yerr=std_intensities, capsize=5, alpha=0.7)
        ax1.set_title("Mean Signal Intensity by Serotype")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.grid(True, alpha=0.3)

        # 1.2 SNR Comparison
        ax2 = axes[0, 1]
        mean_snr = [snr_analysis[s]["mean_snr"] for s in serotypes if s in snr_analysis]
        std_snr = [snr_analysis[s]["std_snr"] for s in serotypes if s in snr_analysis]
        snr_serotypes = [s for s in serotypes if s in snr_analysis]

        ax2.bar(snr_serotypes, mean_snr, yerr=std_snr, capsize=5, alpha=0.7, color="orange")
        ax2.set_title("Signal-to-Noise Ratio by Serotype")
        ax2.set_ylabel("SNR")
        ax2.grid(True, alpha=0.3)

        # 1.3 Reproducibility
        ax3 = axes[1, 0]
        mean_cv = [reproducibility[s]["mean_cv_percent"] for s in serotypes if s in reproducibility]
        repro_serotypes = [s for s in serotypes if s in reproducibility]

        ax3.bar(repro_serotypes, mean_cv, alpha=0.7, color="green")
        ax3.set_title("Reproducibility (CV%) by Serotype")
        ax3.set_ylabel("Coefficient of Variation (%)")
        ax3.grid(True, alpha=0.3)

        # 1.4 Sample Count
        ax4 = axes[1, 1]
        sample_counts = [signal_quality[s]["n_samples"] for s in serotypes]

        ax4.bar(serotypes, sample_counts, alpha=0.7, color="purple")
        ax4.set_title("Number of Samples by Serotype")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/experiment_1_results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_experiment_2_results(self, conc_response, detection_limits, linearity, sensitivity):
        """Generate visualizations for Experiment 2"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Experiment 2: Concentration-Dependent Analysis", fontsize=16, fontweight="bold"
        )

        # 2.1 Concentration-Response Curves
        ax1 = axes[0, 0]
        colors = ["blue", "red"]

        for i, serotype in enumerate(["ST", "SE"]):
            if serotype in conc_response:
                serotype_data = [
                    entry for entry in self.data_entries if entry["serotype"] == serotype
                ]
                concentrations = []
                intensities = []

                for entry in serotype_data:
                    for j, conc in enumerate(entry["concentrations"]):
                        signal = entry["signals"][:, j]
                        max_intensity = np.max(signal)
                        concentrations.append(conc)
                        intensities.append(max_intensity)

                # Plot scatter
                ax1.scatter(concentrations, intensities, alpha=0.6, color=colors[i], label=serotype)

                # Plot trend line
                if len(concentrations) > 1:
                    log_conc = np.log10(concentrations)
                    z = np.polyfit(log_conc, intensities, 1)
                    p = np.poly1d(z)
                    x_trend = np.logspace(
                        np.log10(min(concentrations)), np.log10(max(concentrations)), 100
                    )
                    y_trend = p(np.log10(x_trend))
                    ax1.plot(x_trend, y_trend, "--", color=colors[i], alpha=0.8)

        ax1.set_xscale("log")
        ax1.set_xlabel("Concentration (CFU/ml)")
        ax1.set_ylabel("Max Signal Intensity")
        ax1.set_title("Concentration-Response Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2.2 Detection Limits
        ax2 = axes[0, 1]
        serotypes = list(detection_limits.keys())
        lod_values = [detection_limits[s]["lod"] for s in serotypes]
        loq_values = [detection_limits[s]["loq"] for s in serotypes]

        x = np.arange(len(serotypes))
        width = 0.35

        ax2.bar(x - width / 2, lod_values, width, label="LOD", alpha=0.7)
        ax2.bar(x + width / 2, loq_values, width, label="LOQ", alpha=0.7)
        ax2.set_xlabel("Serotype")
        ax2.set_ylabel("Intensity (a.u.)")
        ax2.set_title("Detection Limits")
        ax2.set_xticks(x)
        ax2.set_xticklabels(serotypes)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 2.3 Linearity (R²)
        ax3 = axes[1, 0]
        serotypes = list(linearity.keys())
        r_squared_values = [linearity[s]["r_squared"] for s in serotypes]

        bars = ax3.bar(serotypes, r_squared_values, alpha=0.7, color="green")
        ax3.axhline(y=0.9, color="red", linestyle="--", label="Acceptable Linearity (R² > 0.9)")
        ax3.set_ylabel("R²")
        ax3.set_title("Linearity Assessment")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, r_squared_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # 2.4 Sensitivity
        ax4 = axes[1, 1]
        serotypes = list(sensitivity.keys())
        sensitivity_values = [sensitivity[s]["sensitivity"] for s in serotypes]

        ax4.bar(serotypes, sensitivity_values, alpha=0.7, color="orange")
        ax4.set_ylabel("Sensitivity (Intensity/log10(CFU))")
        ax4.set_title("Sensor Sensitivity")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/experiment_2_results.png", dpi=300, bbox_inches="tight")
        plt.show()

    def generate_protocol_report(self):
        """Generate comprehensive protocol report"""
        print("\n" + "=" * 60)
        print("📋 GENERATING PROTOCOL REPORT")
        print("=" * 60)

        report = {
            "protocol_info": {
                "title": "Chicken SERS Sensor Testing Protocol",
                "date": datetime.now().isoformat(),
                "data_folder": self.data_folder,
                "total_samples": len(self.data_entries),
            },
            "experiments": self.results,
            "summary": self._generate_summary(),
        }

        # Save report as JSON
        import json

        with open(f"{self.output_dir}/protocol_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate text summary
        self._print_summary_report(report)

        return report

    def _generate_summary(self):
        """Generate summary statistics"""
        summary = {
            "total_experiments": len(self.results),
            "data_quality": "Good" if len(self.data_entries) > 100 else "Limited",
            "serotypes_analyzed": list(set(entry["serotype"] for entry in self.data_entries)),
            "concentration_range": self._get_concentration_range(),
            "sensor_count": len(set(entry["sensor_id"] for entry in self.data_entries)),
        }
        return summary

    def _get_concentration_range(self):
        """Get concentration range from data"""
        all_concentrations = []
        for entry in self.data_entries:
            all_concentrations.extend(entry["concentrations"])

        return [min(all_concentrations), max(all_concentrations)] if all_concentrations else [0, 0]

    def _print_summary_report(self, report):
        """Print summary report to console"""
        print("\n" + "=" * 60)
        print("📊 PROTOCOL ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"📅 Date: {report['protocol_info']['date']}")
        print(f"📁 Data Folder: {report['protocol_info']['data_folder']}")
        print(f"📊 Total Samples: {report['protocol_info']['total_samples']}")
        print(f"🧪 Experiments Completed: {report['summary']['total_experiments']}")
        print(f"🦠 Serotypes Analyzed: {', '.join(report['summary']['serotypes_analyzed'])}")
        print(
            f"📈 Concentration Range: {report['summary']['concentration_range'][0]:.1f} - {report['summary']['concentration_range'][1]:.1f} CFU/ml"
        )
        print(f"🔧 Sensors Used: {report['summary']['sensor_count']}")

        print(f"\n📂 Results saved to: {self.output_dir}/")
        print("   - experiment_1_results.png")
        print("   - experiment_2_results.png")
        print("   - protocol_report.json")


def main():
    """
    Main function to run the chicken protocol analysis
    """
    print("🐔 Chicken SERS Sensor Testing Protocol Analysis")
    print("=" * 60)

    # Initialize analyzer
    data_folder = "../data/SERS Data 7 (Mar 2025)/"
    analyzer = ChickenProtocolAnalyzer(data_folder)

    # Load data
    analyzer.load_chicken_data()

    # Run experiments
    print("\n🚀 Starting Protocol Experiments...")

    # Experiment 1: Basic Analysis
    analyzer.experiment_1_basic_analysis()

    # Experiment 2: Concentration Analysis
    analyzer.experiment_2_concentration_analysis()

    # Generate final report
    analyzer.generate_protocol_report()

    print("\n✅ Protocol analysis completed successfully!")
    print("📋 Check the 'protocol_results' folder for detailed results.")


if __name__ == "__main__":
    main()
