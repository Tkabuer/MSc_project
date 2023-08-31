import numpy as np
import argparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

def main(args):
    print('begin')
    feat = np.load(args.feat_path, allow_pickle=True)
    scores = np.load(args.ground_truth_path, allow_pickle=True)
    
    mse_scores = []  # To store MSE for each iteration
    cnt = 0
    for _ in range(1):
        # Split data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(feat, scores, test_size=0.2)

        # Train regression
        reg = Ridge(alpha=args.alpha).fit(X_train, y_train)

        # Predict on test data and compute MSE
        y_pred = reg.predict(X_test)
        print('x_test:', X_test)
        print('y_pred:', y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        cnt += 1
        if cnt % 100 == 0:
            # Average MSE over all iterations
            avg_mse = np.mean(mse_scores)
            print(f"Average MSE over {cnt} iterations: {avg_mse}")
    
    # Optional: Save the last model
    pickle.dump(reg, open('lin_regressor_bvi.save','wb'))

def parse_args():
    parser = argparse.ArgumentParser(description="linear regressor")
    parser.add_argument('--feat_path', type=str, default='BVI_VFI_fea/bvi_fea_60.pkl', help='path to features file')
    parser.add_argument('--ground_truth_path', type=str, default='BVI_VFI_dmos/bvi_score_60.pkl', help='path to ground truth scores')
    parser.add_argument('--alpha', type=float, default=0.1, help='regularization coefficient')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
