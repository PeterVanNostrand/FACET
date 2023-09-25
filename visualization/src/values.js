
export const rect_values = {
    fill: "white",
    round: 5,
    stroke_width: 2,
    stroke: "black",
}

export const expl_colors = {
    desired: "blue",
    undesired: "red",
    altered_bad: "red",
    altered_good: "blue",
    unaltered: "#939393"
}

export const ExplanationTypes = Object.freeze({
    Region: "region",
    Example: "example"
})

export const ExampleGeneration = Object.freeze({
    Nearest: "nearest",
    Centered: "centered"
})

export const FeatureOrders = Object.freeze({
    Dataset: "dataset",
    LargestChange: "largestchange"
})

export const FEATURE_ORDER = FeatureOrders.Dataset
export const OFFSET_UNSCALED = 0.0001;
export const GENERATION_TYPE = ExampleGeneration.Nearest;
export const CURRENCY_DIGITS = 0;