# to store
torch.save({
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
}, 'filename.pth.tar')

# to load
checkpoint = torch.load('filename.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])