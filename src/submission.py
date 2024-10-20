import pandas as pd
import numpy as np

def create_submission(best_model_tuned, df_test_selected, sample_submission, submission_file):
    # Load the sample submission file
    sample_submission = pd.read_csv(sample_submission)

    # Make predictions on the unseen data (test set)
    y_pred_log = best_model_tuned.predict(df_test_selected)
    y_pred_original_scale = np.exp(y_pred_log)

    # Prepare the submission DataFrame
    submission_df = pd.DataFrame({
        # Keep the Ids from the sample submission
        'Id': sample_submission['Id'],
        'SalePrice': y_pred_original_scale  # Use the original scale predictions
    })

    # Save the submission DataFrame to a CSV file
    submission_df.to_csv(submission_file, index=False)

    print(f"Submission file saved to {submission_file}")
