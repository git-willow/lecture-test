import numpy as np

data = np.array([
    [[65, 70], [56, 80], [78, 68], [90, 85], [60, 75]],
    [[70, 75], [54, 88], [82, 64], [88, 83], [58, 78]],
    [[67, 72], [52, 82], [80, 66], [86, 80], [59, 74]]])

print("Data shape:", data.shape)
print("各クラスの科目別平均点:\n", np.mean(data, axis = 1))
print("全クラスの番号3番の学生の中で2科目目の最高得点:", np.max(data[:,2,1]))
print("各科目で一番点数が高い人と低い人の点差:", np.max(data.reshape(15, -1), axis = 0) - np.min(data.reshape(15, -1), axis = 0))
print("各クラスで1科目目が80点以上の人数:", np.sum(data[:,:,0] >= 80, axis = 1))
print("2科目の合計点が135点を超えている人数:", np.sum(np.sum(data, axis = 2) > 135))
print("全生徒の1科目目と2科目目の相関係数:",
        np.sum((data[:,:,0] - np.mean(data[:,:,0])) * (data[:,:,1] - np.mean(data[:,:,1]))) / 
        (np.sqrt(np.sum((data[:,:,0] - np.mean(data[:,:,0])) ** 2)) *
        np.sqrt(np.sum((data[:,:,1] - np.mean(data[:,:,1])) ** 2))))