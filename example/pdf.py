import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PDF:
    # A context manager which helps us avoid having too many figures open at once.
    #
    # import matplotlib.pyplot as plt
    # with PDF('someplace-to-draw-figures.pdf) as pdf:
    #   for (a lot of iterations):
    #       ... construct some plt fig ...
    #       pdf.save(fig)
    #       plt.close(fig)

    def __init__(self, filename):
        self.filename = filename
        self.pages = None

    def save(self, fig):
        if self.pages:
            fig.savefig(self.pages, format='pdf')
            plt.close(fig)

    def __enter__(self):
        if self.filename:
            self.pages = PdfPages(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.pages:
            self.pages.close()
            self.pages = None
