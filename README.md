# Deep Learning for Tropospheric Ozone Predictions
*November 15th, 2018*

The POLYTECH MTD is...

## Getting Started
Necessary materials

### Prerequisites
What things you need to do before installing and executing the software.

## Installation
The following installation procedures will give a comprehensive walkthrough depending on your operating system

### Linux/Unix
1. Install and update packages
```bash
apt-get update --fix-missing -y
apt-get dist-upgrade -y
apt-get autoremove -y
apt-get autoclean -y
```
2. Install additional packages (if not already installed)
```bash
apt-get install python3 wget
```
3. Install Python packages
```bash
pip install --upgrade tensorflow numpy pandas sodapy
pip3 install --upgrade tensorflow numpy pandas sodapy
```
4. Clone the repository: `git clone https://github.com/computer-geek64/MTD`
5. Execute `Main.py`

### Windows
1. Download latest Python 3.6.* ([Python 3.6.7](https://www.python.org/ftp/python/3.6.7/python-3.6.7-amd64.exe))
2. Install Python 3.6, and make sure to install pip as well
3. Install Python packages
```
pip install --upgrade tensorflow numpy pandas sodapy
pip3 install --upgrade tensorflow numpy pandas sodapy
```
4. Download the [repository](https://github.com/computer-geek64/MTD/archive/master.zip)
5. Extract the repository
6. Execute `Main.py`

### Mac OS X
1. Needs updating

### Deployment
* When on Linux/Unix/Mac OS X, do not execute the program by running `./Main.py`. Instead, run `python3 Main.py` to ensure all packages are present.
* Always ensure that the line endings are compatible with the operating system that you are using
* If a warning message surfaces regarding `pip`, execute the following command:
  * Windows: `python -m pip install --upgrade pip`.
  * Linux/Unix/Mac OS X: `python -m pip install --upgrade pip; python3 -m pip install --upgrade pip`

## Execution
* After

### Functionality
Explain each function of the accessible virtual keyboard

## Built With
* Software:
  * [Python](https://www.python.org/) - *Primary project language*
    * [TensorFlow](https://www.tensorflow.org/) - *Machine learning back-end development*
    * [NumPy](http://www.numpy.org/) - *Scientific computation package*
    * [SodaPy](https://pypi.org/project/sodapy/) - *Socrata Open Data API for Python*
    * [Pandas](https://pandas.pydata.org/) - *Efficient data structure and analysis tools*

## Contributing
Please read the [CONTRIBUTING.md](/docs/CONTRIBUTING.md) file for details on my code of conduct and pull request policy.

## Versioning
This project uses [git](https://git-scm.com/) version control.

## Developers
* **Ashish D'Souza** - *Sole full-stack developer* - [computer-geek64](https://github.com/computer-geek64/)

See also the list of [contributors](/docs/CONTRIBUTORS.md) who participated in this project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for detaills.

## Acknowledgements
* **Mr. Bogdziewicz** for giving me help with my MTD whenever I needed it
* **Mr. Watson** for giving me help with my MTD whenever I needed it
* **Mrs. Mello** for conducting a thorough analysis of my MTD paper
