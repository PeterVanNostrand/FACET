
export const rect_values = {
    fill: "white",
    round: 5,
    stroke_width: 2,
    stroke: "black",
}

export const expl_colors = {
    desired: "#006eff",
    undesired: "#e73b3c",
    altered_bad: "#e73b3c",
    altered_good: "#006eff",
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