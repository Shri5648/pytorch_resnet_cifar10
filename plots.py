def plot_data(train_prec1_history, val_prec1_history):

    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_prec1_history, label='Training Top-1 Acc')
    plt.plot(epochs, val_prec1_history, label='Validation Top-1 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Training vs. Validation Top-1 Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png')    # write to file
end
