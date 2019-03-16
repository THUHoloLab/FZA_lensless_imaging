function sol=diffv(x)
h=[0 1 -1]';
sol=conv2c(x,h);