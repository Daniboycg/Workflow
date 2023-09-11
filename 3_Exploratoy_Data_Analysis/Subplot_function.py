# Definiendo Función
def plot_categoricals(df, colums, ncols=2):
    # Numero de graficas a realizar
    nplot = len(colums)
    nrows = (nplot // ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 30), constrained_layout=True)
    for row in range(nrows):
        for col in range(ncols):
            title = colums[row + col]
            sns.countplot(data=df, x=title, hue="Churn", ax=axes[row, col])
            axes[row, col].set_title("COUNT " + title.upper())
            axes[row, col].set_xlabel(title)
            axes[row, col].set_ylabel("Count")
            axes[row, col].legend()
    fig.tight_layout()
    plt.show()


# Graficando con la Función
plot_categoricals(column_categorical, 3)
