from pathlib import Path
import csv

# fft 2D d and di
def is_fft_2d(item):
    if "ft2d" in item.name and "-f_" in item.name and "-B--" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_1D_f_16384_16384", line.split(" ")[-1].rstrip('\n\r')]

    if "ft2d" in item.name and "-fb_" in item.name and "-B--" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_1D_fb_16384_16384", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

# fft 2D batch d and di
def is_fft_2d_batch(item):
    if "ft2d" in item.name and "-f-" in item.name and "-batch_" in item.name and "-B-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_f_batch_1024_1024_512", line.split(" ")[-1].rstrip('\n\r')]

    if "ft2d" in item.name and "-fb-" in item.name and "-batch_" in item.name and "-B-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_fb_batch_1024_1024_512", line.split(" ")[-1].rstrip('\n\r')]
    
    if "ft2d" in item.name and "-f-" in item.name and "-batch_" in item.name and "-X-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_f_batch_4096_4096_32", line.split(" ")[-1].rstrip('\n\r')]

    if "ft2d" in item.name and "-fb-" in item.name and "-batch_" in item.name and "-X-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_fb_batch_4096_4096_32", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

# fft 2D batch d and di + mm
def is_fft_2d_mm(item):
    if "ft2d" in item.name and "-f-" in item.name and "-mm_" in item.name and "-B-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_f_batch_gemm_1024_1024_512", line.split(" ")[-1].rstrip('\n\r')]

    if "ft2d" in item.name and "-fb-" in item.name and "-mm_" in item.name and "-B-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_fb_batch_gemm_1024_1024_512", line.split(" ")[-1].rstrip('\n\r')]
    
    if "ft2d" in item.name and "-f-" in item.name and "-mm_" in item.name and "-X-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_f_batch_gemm_4096_4096_32", line.split(" ")[-1].rstrip('\n\r')]

    if "ft2d" in item.name and "-fb-" in item.name and "-mm_" in item.name and "-X-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft2D_2D_fb_batch_gemm_4096_4096_32", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

# fft 3D di
def is_fft_3d(item):
    if "ft3d" in item.name and "fb_1D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_1D_fb_256_256_256", line.split(" ")[-1].rstrip('\n\r')]

    if "ft3d" in item.name and "fb_2D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_2D_fb_256_256_256", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

# fft 3D batch di
def is_fft_3d_batch(item):
    if "ft3d" in item.name and "fb-batch_2D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_2D_fb_batch_256_256_256_1024", line.split(" ")[-1].rstrip('\n\r')]

    if "ft3d" in item.name and "fb-batch_3D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_3D_fb_batch_256_256_256_1024", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

# fft 3D batch di + mm
def is_fft_3d_mm(item):
    if "ft3d" in item.name and "fb-mm_2D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
        
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_2D_fb_batch_gemm_256_256_256_1024", line.split(" ")[-1].rstrip('\n\r')]

    if "ft3d" in item.name and "fb-mm_3D-" in item.name:
        filename = item.parent.name + "/" + item.name + "/current.sum"
    
        with open(filename, 'r') as file:
            for line in file:
                if "Search Key" in line:
                    return ["fft3D_3D_fb_batch_gemm_256_256_256_1024", line.split(" ")[-1].rstrip('\n\r')]
    
    return []

folder_path = Path("../solver/solutions/") # Replace with your folder's path

fft_2D_data = []
fft_2D_batch_data = []
fft_2D_mm_data = []

fft_3D_data = []
fft_3D_batch_data = []
fft_3D_mm_data = []

for item in folder_path.iterdir():
    result = is_fft_2d(item)
    if(len(result) != 0):
        fft_2D_data.append(result)
        
    result = is_fft_2d_batch(item)
    if(len(result) != 0):
        fft_2D_batch_data.append(result)
    
    result = is_fft_2d_mm(item)
    if(len(result) != 0):
        fft_2D_mm_data.append(result)
        
    result = is_fft_3d(item)
    if(len(result) != 0):
        fft_3D_data.append(result)
        
    result = is_fft_3d_batch(item)
    if(len(result) != 0):
        fft_3D_batch_data.append(result)
    
    result = is_fft_3d_mm(item)
    if(len(result) != 0):
        fft_3D_mm_data.append(result)

with open("fft_2D/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_2D_data)
    
with open("fft_2D_batch/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_2D_batch_data)
    
with open("fft_2D_batch_mm/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_2D_mm_data)
    
with open("fft_3D/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_3D_data)
    
with open("fft_3D_batch/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_3D_batch_data)
    
with open("fft_3D_batch_mm/result_new.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|') 
    writer.writerows(fft_3D_mm_data)