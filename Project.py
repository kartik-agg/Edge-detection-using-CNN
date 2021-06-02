import matplotlib
import numpy
import skimage.data
import CNN as numpycnn
import matplotlib.pyplot


# Reading the image
# Can uncomment any line 
#img = skimage.data.checkerboard()
img = skimage.data.camera()
#img = skimage.data.coffee()
#img = skimage.data.chelsea()


# Converting the image into gray.
img = skimage.color.rgb2gray(img)

# First conv layer
l1_filter = numpy.zeros((2,3,3))
for i in range(2):
    
    l1_filter[i, :, :] = numpy.array([[[-1, 0, 1], 
                                   [-2, 0, 2], 
                                   [-1, 0, 1]]])     #Sobel Filter for vertical lines detection

print("\n**Working with conv layer 1**")
l1_feature_map = numpycnn.conv(img, l1_filter)
print("\n**ReLU**")
l1_feature_map_relu = numpycnn.relu(l1_feature_map)
print("\n**Pooling**")
l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map_relu, 2, 2)
print("**End of conv layer 1**\n")

# Second conv layer
l2_filter = numpy.zeros((2,3,3))
#Filter in second conv layer has 2 no. of channels
for i in range(2):
    
    l2_filter[i, :, :] = numpy.array([[[-1, -2, -1], 
                                        [0, 0, 0], 
                                        [1, 2, 1]]])  #Sobel Filter for horizontal lines detection
        
print("\n**Working with conv layer 2**")
l2_feature_map = numpycnn.conv(img, l2_filter)
print("\n**ReLU**")
l2_feature_map_relu = numpycnn.relu(l2_feature_map)
print("\n**Pooling**")
l2_feature_map_relu_pool = numpycnn.pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**\n")


# Third conv layer
l3_filter = numpy.zeros((2,3,3))
#Filter in third conv layer has 2 no. of channels
for i in range(2):
    
    l3_filter[i, :, :] = numpy.array([[[0, 1, 2], 
                                    [-1, 0, 1], 
                                    [-2, -1, 0]]])  #Sobel Filter for 45 degree lines detection
        
print("\n**Working with conv layer 3**")
l3_feature_map = numpycnn.conv(img, l3_filter)
print("\n**ReLU**")
l3_feature_map_relu = numpycnn.relu(l3_feature_map)
print("\n**Pooling**")
l3_feature_map_relu_pool = numpycnn.pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")


# Fourth conv layer
l4_filter = numpy.zeros((2,3,3))
#Filter in fourth conv layer has 2 no. of channels
for i in range(2):
    
    l4_filter[i, :, :] = numpy.array([[[-2, -1, 0], 
                                    [-1, 0, 1], 
                                    [0, 1, 2]]])  #Sobel Filter for 135 degree lines detection
        
print("\n**Working with conv layer 4**")
l4_feature_map = numpycnn.conv(img, l4_filter)
print("\n**ReLU**")
l4_feature_map_relu = numpycnn.relu(l4_feature_map)
print("\n**Pooling**")
l4_feature_map_relu_pool = numpycnn.pooling(l4_feature_map_relu, 2, 2)
print("**End of conv layer 4**\n")




# Fifth conv layer
l5_filter = numpy.zeros((2,3,3))
#Filter in fourth conv layer has 2 no. of channels
for i in range(2):
    
    l5_filter[i, :, :] = numpy.array([[[0, 1, 0], 
                                    [1, -4, 1], 
                                    [0, 1, 0]]])  #Filter for total edge detection
        
print("\n**Working with conv layer 5**")
l5_feature_map = numpycnn.conv(img, l5_filter)

print("\n**ReLU**")
l5_feature_map_relu = numpycnn.relu(l5_feature_map)
print("\n**Pooling**")
l5_feature_map_relu_pool = numpycnn.pooling(l5_feature_map_relu, 2, 2)
print("**End of conv layer 5**\n")












print('*********************************************************************')
















# Graphing results
fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
matplotlib.pyplot.savefig("in_img.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig0)

# Layer 1
fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(l1_feature_map[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(l1_feature_map[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")

ax1[1, 0].imshow(l1_feature_map_relu[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(l1_feature_map_relu[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")

ax1[2, 0].imshow(l1_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(l1_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")

matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig1)

# Layer 2
fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax2[0, 0].imshow(l2_feature_map[:, :, 0]).set_cmap("gray")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])
ax2[0, 0].set_title("L2-Map1")

ax2[0, 1].imshow(l2_feature_map[:, :, 1]).set_cmap("gray")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])
ax2[0, 1].set_title("L2-Map2")


ax2[1, 0].imshow(l2_feature_map_relu[:, :, 0]).set_cmap("gray")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])
ax2[1, 0].set_title("L2-Map1ReLU")

ax2[1, 1].imshow(l2_feature_map_relu[:, :, 1]).set_cmap("gray")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])
ax2[1, 1].set_title("L2-Map2ReLU")



ax2[2, 0].imshow(l2_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])
ax2[2, 0].set_title("L2-Map1ReLUPool")

ax2[2, 1].imshow(l2_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])
ax2[2, 1].set_title("L2-Map2ReLUPool")



matplotlib.pyplot.savefig("L2.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig2)




# Layer 3
fig3, ax3 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax3[0, 0].imshow(l3_feature_map[:, :, 0]).set_cmap("gray")
ax3[0, 0].get_xaxis().set_ticks([])
ax3[0, 0].get_yaxis().set_ticks([])
ax3[0, 0].set_title("L3-Map1")

ax3[0, 1].imshow(l3_feature_map[:, :, 1]).set_cmap("gray")
ax3[0, 1].get_xaxis().set_ticks([])
ax3[0, 1].get_yaxis().set_ticks([])
ax3[0, 1].set_title("L3-Map2")


ax3[1, 0].imshow(l3_feature_map_relu[:, :, 0]).set_cmap("gray")
ax3[1, 0].get_xaxis().set_ticks([])
ax3[1, 0].get_yaxis().set_ticks([])
ax3[1, 0].set_title("L3-Map1ReLU")

ax3[1, 1].imshow(l3_feature_map_relu[:, :, 1]).set_cmap("gray")
ax3[1, 1].get_xaxis().set_ticks([])
ax3[1, 1].get_yaxis().set_ticks([])
ax3[1, 1].set_title("L3-Map2ReLU")



ax3[2, 0].imshow(l3_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax3[2, 0].get_xaxis().set_ticks([])
ax3[2, 0].get_yaxis().set_ticks([])
ax3[2, 0].set_title("L3-Map1ReLUPool")

ax3[2, 1].imshow(l3_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax3[2, 1].get_xaxis().set_ticks([])
ax3[2, 1].get_yaxis().set_ticks([])
ax3[2, 1].set_title("L3-Map2ReLUPool")



matplotlib.pyplot.savefig("L3.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig3)



# Layer 4
fig4, ax4 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax4[0, 0].imshow(l4_feature_map[:, :, 0]).set_cmap("gray")
ax4[0, 0].get_xaxis().set_ticks([])
ax4[0, 0].get_yaxis().set_ticks([])
ax4[0, 0].set_title("L4-Map1")

ax4[0, 1].imshow(l4_feature_map[:, :, 1]).set_cmap("gray")
ax4[0, 1].get_xaxis().set_ticks([])
ax4[0, 1].get_yaxis().set_ticks([])
ax4[0, 1].set_title("L4-Map2")


ax4[1, 0].imshow(l4_feature_map_relu[:, :, 0]).set_cmap("gray")
ax4[1, 0].get_xaxis().set_ticks([])
ax4[1, 0].get_yaxis().set_ticks([])
ax4[1, 0].set_title("L4-Map1ReLU")

ax4[1, 1].imshow(l4_feature_map_relu[:, :, 1]).set_cmap("gray")
ax4[1, 1].get_xaxis().set_ticks([])
ax4[1, 1].get_yaxis().set_ticks([])
ax4[1, 1].set_title("L4-Map2ReLU")



ax4[2, 0].imshow(l4_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax4[2, 0].get_xaxis().set_ticks([])
ax4[2, 0].get_yaxis().set_ticks([])
ax4[2, 0].set_title("L4-Map1ReLUPool")

ax4[2, 1].imshow(l4_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax4[2, 1].get_xaxis().set_ticks([])
ax4[2, 1].get_yaxis().set_ticks([])
ax4[2, 1].set_title("L4-Map2ReLUPool")



matplotlib.pyplot.savefig("L4.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig4)




# Final Result
fig5, ax5 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax5[0, 0].imshow(l5_feature_map[:, :, 0]).set_cmap("gray")
ax5[0, 0].get_xaxis().set_ticks([])
ax5[0, 0].get_yaxis().set_ticks([])
ax5[0, 0].set_title("L5-Map1")

ax5[0, 1].imshow(l5_feature_map[:, :, 1]).set_cmap("gray")
ax5[0, 1].get_xaxis().set_ticks([])
ax5[0, 1].get_yaxis().set_ticks([])
ax5[0, 1].set_title("L5-Map2")

ax5[1, 0].imshow(l5_feature_map_relu[:, :, 0]).set_cmap("gray")
ax5[1, 0].get_xaxis().set_ticks([])
ax5[1, 0].get_yaxis().set_ticks([])
ax5[1, 0].set_title("L5-Map1RelU")

ax5[1, 1].imshow(l5_feature_map_relu[:, :, 1]).set_cmap("gray")
ax5[1, 1].get_xaxis().set_ticks([])
ax5[1, 1].get_yaxis().set_ticks([])
ax5[1, 1].set_title("L5-Map2RelU")

ax5[2, 0].imshow(l5_feature_map_relu_pool[:, :, 0]).set_cmap("gray")
ax5[2, 0].get_xaxis().set_ticks([])
ax5[2, 0].get_yaxis().set_ticks([])
ax5[2, 0].set_title("L5-Map1RelUPool")

ax5[2, 1].imshow(l5_feature_map_relu_pool[:, :, 1]).set_cmap("gray")
ax5[2, 1].get_xaxis().set_ticks([])
ax5[2, 1].get_yaxis().set_ticks([])
ax5[2, 1].set_title("L5-Map2RelUPool")

matplotlib.pyplot.savefig("Final Image.png", bbox_inches="tight")
matplotlib.pyplot.show()
matplotlib.pyplot.close(fig5)














