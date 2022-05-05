import os


offset = 2
num_img = 27
for i in range(offset, num_img+1):
	old_name = "./data/calib_image_fish_" + str(i) + ".png"
	new_name = "./data/calib_image_fish_" + str(i-offset) + ".png"
	os.rename(old_name, new_name)

