
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import numpy as np
from PIL import Image
#tk for drawing pictures
img = Image.open("slika.jpg")

a = np.array(img)
plt.imshow(a)
plt.show()
greyimg= img.convert("L")


# In[6]:


b = np.array(greyimg)


# In[8]:


plt.imshow(b,cmap="gray")


# In[9]:


greyimg.save("gray.png")

