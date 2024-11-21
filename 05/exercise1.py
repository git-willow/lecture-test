from pathlib import Path

print("===== problem1 =====")
base_dirctory = Path("./data")
base_dirctory_path = base_dirctory.resolve()
print(base_dirctory_path)

print("===== problem2 =====")
for file in list(base_dirctory_path.glob("*")):
    print(file)

print("===== problem3 =====")
count = 0
for file in list(base_dirctory_path.glob("*")):
    for image in list(file.glob("*.png")):
        count += 1
print(count)
