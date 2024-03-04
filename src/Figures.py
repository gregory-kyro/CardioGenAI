# Import necessary libraries
from IPython.display import display

from src.Dataset_Analysis import (
    analyze_cardiac_ion_channel_dataset,
    analyze_top_and_bottom_pIC50_compounds,
)
from src.Discriminator import (
    evaluate_discriminator_model,
    plot_roc_curve_for_each_cardiac_ion_channel,
    screen_FDA_compounds,
)
from src.Discriminator_Feature_Selection import (
    evaluate_feature_variable_discriminator_models,
    plot_roc_curve_for_different_feature_combinations,
)
from src.Visualization import (
    analyze_FDA_compound_predictions,
    plot_attention_weights_for_both_models,
    plot_learning_curves,
    plot_pca_space,
    plot_pIC50s_of_generated_compounds,
)


# Define a function to get all figures in the manuscript
def get_figures():
    """
    Generates and returns various figures related to the CardioGenAI project.

    This function creates both main text figures and supplementary figures by calling
    different functions. The main text figures include evaluating the discriminator model,
    screening FDA compounds, and plotting PCA space. The supplementary figures include
    analyzing the cardiac ion channel dataset, analyzing top and bottom pIC50 compounds,
    plotting learning curves, evaluating feature variable discriminator models, plotting
    ROC curve for different feature combinations, plotting ROC curve for each cardiac ion
    channel, evaluating the discriminator model for regression prediction, analyzing FDA
    compound predictions, plotting attention weights for both models, and plotting pIC50s
    of generated compounds.

    Returns:
        None
    """

    # Main text figures
    discriminator_evaluation = evaluate_discriminator_model()
    print("Discriminator Classification Evaluation:")
    print(discriminator_evaluation.to_string())
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    fda_screening_results = screen_FDA_compounds()
    print("FDA Screening Results:")
    print(fda_screening_results.to_string())
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("PCA Space:")
    plot_pca_space()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    # Supplementary figures
    print("hERG Dataset Analysis:")
    analyze_cardiac_ion_channel_dataset()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    pic50_analysis_image = analyze_top_and_bottom_pIC50_compounds()
    print("hERG pIC50 Analysis:")
    display(pic50_analysis_image)
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("Learning Curves:")
    plot_learning_curves()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    feature_evaluation = evaluate_feature_variable_discriminator_models()
    print("Feature Representation Evaluation:")
    print(feature_evaluation)
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("ROC Curve for Different Feature Combinations:")
    plot_roc_curve_for_different_feature_combinations()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("ROC Curve for Each Cardiac Ion Channel:")
    plot_roc_curve_for_each_cardiac_ion_channel()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("Discriminator Regression Evaluation:")
    evaluate_discriminator_model(prediction_type="regression")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("FDA Compound Prediction Analysis:")
    analyze_FDA_compound_predictions()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("Attention Weights for Both Models:")
    plot_attention_weights_for_both_models()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    print("Generated Compound pIC50s:")
    plot_pIC50s_of_generated_compounds()
