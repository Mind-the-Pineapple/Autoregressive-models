out[0, 0, 5, 5, 5].backward()
grad = x.grad.detach().cpu().numpy()[0, 0]
grad = np.abs(grad)
grad = (grad != 0).astype('float64')
grad[5, 5, 5] = 0.5
from  matplotlib.ticker import FixedLocator
plt.figure()
for i in range(10):
    axes = plt.subplot(2,5,i+1)
    plt.title(f"out[{i},:,:]")
    plt.xticks(np.arange(0, 10, step=1))
    plt.yticks(np.arange(0, 10, step=1))
    axes.xaxis.set_minor_locator(FixedLocator(np.arange(0.5, 10.5, step=1)))
    axes.yaxis.set_minor_locator(FixedLocator(np.arange(0.5, 10.5, step=1)))
    plt.grid(which="minor")
    plt.xlabel("out[i,:,k]")
    plt.ylabel("out[i,j,:]")
    plt.imshow(grad[i,:,:],vmin=0, vmax=1)

plt.show()
