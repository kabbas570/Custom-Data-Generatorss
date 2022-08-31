size=0
s1=a1[0].element_size()*a1[0].nelement()
s2=a1[1].element_size()*a1[1].nelement()
s3=a1[2].element_size()*a1[2].nelement()
s4=a1[3].element_size()*a1[3].nelement()
s5=a1[4].element_size()*a1[4].nelement()

size1=s1+s2+s3+s4+s5

size2=s1+s2+s3+s4

size3=s1+s2+s3


# img=a1[2].numpy()
# import sys
# sys.getsizeof(img)

# o=a1[4].numpy()

# print("%d bytes" % (img.size * img.itemsize))

# print("%d bytes" % (a1[0].numpy().size * a1[0].numpy().itemsize))
