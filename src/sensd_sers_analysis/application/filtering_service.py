"""
Application-layer filtering services for Streamlit orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sensd_sers_analysis.application.contracts import (
    DerivedDataBundle,
    FilterCatalog,
    FilterSelection,
    FilteredBundle,
)
from sensd_sers_analysis.data import count_unique_spectra
from sensd_sers_analysis.processing import (
    filter_sers_data,
    get_filter_options,
    get_filterable_columns,
)


FilterState = dict[str, FilterSelection]


def build_filter_catalog(
    tidy_df,
    *,
    main_filter_count: int,
) -> FilterCatalog:
    """
    Build the ordered filter catalog used by the sidebar UI.

    Parameters
    ----------
    tidy_df:
        Tidy dataframe used for filter discovery.
    main_filter_count:
        Number of filters to show in the main sidebar section.

    Returns
    -------
    FilterCatalog
        Ordered filter catalog for the sidebar.
    """

    filter_columns = tuple(get_filterable_columns(tidy_df))
    return FilterCatalog(
        filter_columns=filter_columns,
        main_columns=filter_columns[:main_filter_count],
        more_columns=filter_columns[main_filter_count:],
    )


def normalize_filter_state(
    filter_state: Mapping[str, FilterSelection],
) -> dict[str, tuple[list[str] | None, bool]]:
    """
    Convert the application-layer filter state to the processing-layer format.

    Parameters
    ----------
    filter_state:
        Mapping of column name to typed filter selection.

    Returns
    -------
    dict[str, tuple[list[str] | None, bool]]
        Legacy processing-layer state format.
    """

    return {column: selection.as_processing_state() for column, selection in filter_state.items()}


def serialize_filter_state(
    filter_state: Mapping[str, FilterSelection],
) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
    """
    Serialize a filter state for caching.

    Parameters
    ----------
    filter_state:
        Mapping of canonical column names to filter selections.

    Returns
    -------
    tuple[tuple[str, tuple[str, ...], bool], ...]
        Stable serialized representation of the filter state.
    """

    return tuple(
        sorted(
            (
                column,
                selection.selected_values,
                selection.exclude,
            )
            for column, selection in filter_state.items()
        )
    )


def deserialize_filter_state(
    serialized_state: Sequence[tuple[str, tuple[str, ...], bool]],
) -> FilterState:
    """
    Deserialize a cached filter-state representation.

    Parameters
    ----------
    serialized_state:
        Serialized filter state returned by `serialize_filter_state`.

    Returns
    -------
    FilterState
        Typed filter-state mapping.
    """

    return {
        column: FilterSelection(selected_values=selected_values, exclude=exclude)
        for column, selected_values, exclude in serialized_state
    }


def compute_filter_options(
    tidy_df,
    filter_columns: Sequence[str],
    filter_state: Mapping[str, FilterSelection],
) -> dict[str, list]:
    """
    Compute cascading filter options for the current typed filter state.

    Parameters
    ----------
    tidy_df:
        Tidy dataframe used to compute filter options.
    filter_columns:
        Ordered filter columns.
    filter_state:
        Current typed filter-state mapping.

    Returns
    -------
    dict[str, list]
        Available options per filter column.
    """

    return get_filter_options(
        tidy_df,
        list(filter_columns),
        normalize_filter_state(filter_state),
    )


def apply_filters(
    derived_bundle: DerivedDataBundle,
    filter_state: Mapping[str, FilterSelection],
) -> FilteredBundle:
    """
    Apply canonical filters to the derived bundle.

    Parameters
    ----------
    derived_bundle:
        Fully derived application data bundle.
    filter_state:
        Current typed filter-state mapping.

    Returns
    -------
    FilteredBundle
        Filtered tidy/features views and the unique spectra count.
    """

    processing_state = normalize_filter_state(filter_state)
    filtered_tidy_df = filter_sers_data(derived_bundle.tidy_df, processing_state)
    filtered_features_df = filter_sers_data(derived_bundle.features_df, processing_state)
    n_unique_spectra = count_unique_spectra(filtered_tidy_df)
    return FilteredBundle(
        filtered_tidy_df=filtered_tidy_df,
        filtered_features_df=filtered_features_df,
        n_unique_spectra=n_unique_spectra,
    )
