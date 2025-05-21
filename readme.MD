# TF-IDF Test Exercise

This application provides an index page with a form for uploading a text file.
After processing the uploaded file, a page with a table is displayed.

The table consists of the top 50 words and includes the following columns:

1.  **Word**: The word from the text.
2.  **TF (Term Frequency)**: A normalized score representing the word's Term Frequency.
3.  **IDF (Inverse Document Frequency)**: The Inverse Document Frequency score for the word.

The output is ordered by decreasing IDF values. For words with the same IDF, they are secondarily sorted by decreasing TF values.

**Note on IDF:** In this specific implementation, where only a single file is processed at a time (i.e., the document corpus consists of one document), the IDF value for all words present in that document will consistently be **1.0**.