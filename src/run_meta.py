from meta.author import Author
from meta.book import OpenLibrary


if __name__ == "__main__":
    meta = OpenLibrary.query(
        "Faust",
        (Author("Goethe", "Johann Wolfgang von", 1749, 1832),),
    )
    print(meta)
