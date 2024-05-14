import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_beta_distribution_pdf(alpha, save_dir='./diagram', fig_name='beta_distribution_pdf.png'):
    # Create the 'diagram' directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define a range of x values
    x = np.linspace(0, 1, 1000)

    # Calculate the PDF of the beta distribution
    pdf = (x**(alpha-1) * (1-x)**(alpha-1)) / (np.math.gamma(alpha) * np.math.gamma(alpha) / np.math.gamma(2*alpha))

    # Plot the PDF
    fig, ax = plt.subplots()
    ax.plot(x, pdf, label=f'Beta PDF (α={alpha})')
    ax.set_title(f'Beta Distribution PDF (α={alpha})')
    ax.set_xlabel('x')
    ax.set_ylabel('PDF')
    ax.legend()
    ax.grid(True)

    # Save the figure in the specified directory
    plt.savefig(os.path.join(save_dir, fig_name))