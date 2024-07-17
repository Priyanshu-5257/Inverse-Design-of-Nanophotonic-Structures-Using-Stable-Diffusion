# Inverse Design of Nanophotonic Structures Using Stable Diffusion

## Project Overview

This project focuses on the inverse design of nanophotonic structures using Stable Diffusion techniques. The goal is to generate metasurface structures from an absorption spectrum, aiding in the design of advanced optical devices. The project leverages deep learning methodologies, specifically U-Net architecture and denoising diffusion probabilistic models, to achieve high-quality design outputs.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset consists of 20,000 metasurface designs paired with their corresponding absorption spectra. The data is preprocessed and Gaussian filtering is applied to enhance the training process.

## Methodology

### Model Architecture

- **U-Net Architecture:** A convolutional network designed for image segmentation tasks.
- **Denoising Diffusion Probabilistic Models (DDPM):** A framework for training generative models by iteratively denoising data through a Markov chain.

### Training Process

1. **Data Preprocessing:** Applied Gaussian filtering to the dataset to improve model training.
2. **Model Training:** The U-Net model is trained using the methodologies from the DDPM paper.
3. **Evaluation:** The model's performance is evaluated based on its ability to generate coherent metasurface structures from the given spectral data.

## Results

The model demonstrated faster convergence compared to Conditional GANs and successfully generated coherent structures from the spectral data. The use of Stable Diffusion techniques significantly improved the quality of the generated designs.


## Contributing

We welcome contributions to this project. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is based on the methodologies from the Denoising Diffusion Probabilistic Models paper. Special thanks to the team at the Indian Institute of Technology, Roorkee, for their support and guidance.

---

For any questions or issues, please contact [Priyanshu Maurya](mailto:p_maurya@ph.iitr.ac.in).
