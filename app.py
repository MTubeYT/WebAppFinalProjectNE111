import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Histogram Fitting Tool",
    layout="centered"
)

st.title("Histogram Fitting Tool")
st.write(
    "Load some numbers, fit a probability distribution, see the histogram + curve, "
    "and tweak the fit with sliders."
)

# Helpers
def parse_manual_data(text):
    # Turn pasted text into a 1D numpy array of floats
    if not text:
        return np.array([])

    text = text.replace("\n", " ").replace(",", " ")
    parts = [p for p in text.split(" ") if p.strip() != ""]

    numbers = []
    for p in parts:
        try:
            numbers.append(float(p))
        except ValueError:
            # Ignore non-numeric stuff
            pass

    return np.array(numbers)


def get_numeric_columns(df):
    # Return names of numeric columns
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compute_fit_error(dist, params, data, bins=30):
    # Compare histogram to pdf with a simple error metric
    if dist.shapes is not None:
        shapes_count = len(dist.shapes.split(","))
    else:
        shapes_count = 0

    shape_params = params[:shapes_count]
    loc = params[shapes_count]
    scale = params[shapes_count + 1]

    dist_obj = dist(*shape_params, loc=loc, scale=scale)

    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_values = dist_obj.pdf(bin_centers)

    errors = np.abs(counts - pdf_values)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    return mean_error, max_error


def make_plot(dist, params, data, title="Fitted Distribution", bins=30, use_manual=False):
    # Plot histogram and fitted pdf
    if dist.shapes is not None:
        shapes_count = len(dist.shapes.split(","))
    else:
        shapes_count = 0

    shape_params = params[:shapes_count]
    loc = params[shapes_count]
    scale = params[shapes_count + 1]

    dist_obj = dist(*shape_params, loc=loc, scale=scale)

    x_min = float(np.min(data))
    x_max = float(np.max(data))
    x_range = x_max - x_min if x_max > x_min else 1.0

    x_vals = np.linspace(
        x_min - 0.1 * x_range,
        x_max + 0.1 * x_range,
        400
    )
    pdf_vals = dist_obj.pdf(x_vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, density=True, alpha=0.5, edgecolor="black")
    ax.plot(x_vals, pdf_vals, linewidth=2)

    label = "Manual fit" if use_manual else "Automatic fit"
    ax.set_title(f"{title} ({label})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    return fig


# Sidebar
st.sidebar.header("Options")

DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma": stats.gamma,
    "Weibull (weibull_min)": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "Beta": stats.beta,
    "Chi-squared (chi2)": stats.chi2,
    "Uniform": stats.uniform,
    "Student t": stats.t,
    "F-distribution": stats.f,
    "Cauchy": stats.cauchy,
}

dist_name = st.sidebar.selectbox("Choose distribution", list(DISTRIBUTIONS.keys()))
current_dist = DISTRIBUTIONS[dist_name]

num_bins = st.sidebar.slider("Number of histogram bins", 5, 80, 30, 1)
manual_mode = st.sidebar.checkbox("Enable manual fitting (sliders)", value=True)


# Data input
st.header("1. Data Input")

data_source = st.radio(
    "Choose how to provide data:",
    ("Type/paste data", "Upload CSV file"),
    horizontal=True,
)

data = np.array([])

if data_source == "Type/paste data":
    st.write("Enter numbers separated by commas, spaces, or newlines.")
    default_text = "1.2, 2.3, 2.1, 3.5, 3.6, 4.0, 5.2"
    text_data = st.text_area("Data:", value=default_text, height=120)
    data = parse_manual_data(text_data)

else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            numeric_cols = get_numeric_columns(df)
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in CSV file.")
            else:
                col_name = st.selectbox("Select numeric column", numeric_cols)
                column_data = df[col_name].dropna()
                data = column_data.to_numpy()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# Fitting
st.header("2. Distribution Fitting")

if data.size < 5:
    st.info("Provide at least a few numeric data points to perform a fit.")
else:
    st.write(f"Number of data points: **{data.size}**")

    try:
        fitted_params = current_dist.fit(data)
    except Exception as e:
        st.error(f"Could not fit distribution: {e}")
        fitted_params = None

    if fitted_params is not None:
        if current_dist.shapes is not None:
            shape_names = [s.strip() for s in current_dist.shapes.split(",")]
            shapes_count = len(shape_names)
        else:
            shape_names = []
            shapes_count = 0

        st.subheader("Automatic Fit Parameters (from SciPy)")

        auto_param_dict = {}

        for i in range(shapes_count):
            name = shape_names[i] if i < len(shape_names) else f"shape{i+1}"
            auto_param_dict[name] = fitted_params[i]

        auto_param_dict["loc"] = fitted_params[shapes_count]
        auto_param_dict["scale"] = fitted_params[shapes_count + 1]

        st.json({k: float(v) for k, v in auto_param_dict.items()})

        auto_mean_err, auto_max_err = compute_fit_error(
            current_dist, fitted_params, data, bins=num_bins
        )

        st.write("**Fit quality (automatic):**")
        st.write(f"- Mean absolute error: `{auto_mean_err:.4f}`")
        st.write(f"- Max absolute error: `{auto_max_err:.4f}`")

        if manual_mode:
            st.subheader("Manual Fitting (Adjust Sliders)")

            manual_params = list(fitted_params)

            # Shape params
            for i in range(shapes_count):
                name = shape_names[i] if i < len(shape_names) else f"shape{i+1}"
                val = float(fitted_params[i])

                if val == 0:
                    slider_min = 0.0
                    slider_max = 5.0
                else:
                    slider_min = float(val * 0.1)
                    slider_max = float(val * 3.0)
                    if slider_max < slider_min + 1e-6:
                        slider_max = slider_min + 1.0

                manual_params[i] = st.slider(
                    name,
                    min_value=slider_min,
                    max_value=slider_max,
                    value=val,
                )

            # loc
            loc_val = float(fitted_params[shapes_count])
            data_min = float(np.min(data))
            data_max = float(np.max(data))
            loc_min = data_min - (data_max - data_min)
            loc_max = data_max + (data_max - data_min)

            manual_params[shapes_count] = st.slider(
                "loc",
                min_value=loc_min,
                max_value=loc_max,
                value=loc_val,
            )

            # scale
            scale_val = float(fitted_params[shapes_count + 1])
            scale_min = max(scale_val * 0.1, 1e-3)
            if scale_val > 0:
                scale_max = scale_val * 3.0
            else:
                scale_max = 5.0
            if scale_max < scale_min + 1e-6:
                scale_max = scale_min + 1.0

            manual_params[shapes_count + 1] = st.slider(
                "scale",
                min_value=scale_min,
                max_value=scale_max,
                value=scale_val,
            )

            manual_params = tuple(manual_params)

            man_mean_err, man_max_err = compute_fit_error(
                current_dist, manual_params, data, bins=num_bins
            )

            st.write("**Fit quality (manual):**")
            st.write(f"- Mean absolute error: `{man_mean_err:.4f}`")
            st.write(f"- Max absolute error: `{man_max_err:.4f}`")

            view_mode = st.radio(
                "Which curve do you want to see?",
                ("Automatic fit", "Manual fit"),
                horizontal=True,
            )

            if view_mode == "Automatic fit":
                fig = make_plot(
                    current_dist,
                    fitted_params,
                    data,
                    title=dist_name,
                    bins=num_bins,
                    use_manual=False,
                )
            else:
                fig = make_plot(
                    current_dist,
                    manual_params,
                    data,
                    title=dist_name,
                    bins=num_bins,
                    use_manual=True,
                )

            st.pyplot(fig)
            plt.close(fig)

        else:
            fig = make_plot(
                current_dist,
                fitted_params,
                data,
                title=dist_name,
                bins=num_bins,
                use_manual=False,
            )
            st.pyplot(fig)
            plt.close(fig)
