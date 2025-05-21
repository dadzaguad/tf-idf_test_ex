import io
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


N_ROWS_DEFAULT = 50


class TextProcessingError(Exception):
    pass


async def read_and_validate_file_content(
    file: Any, max_size_bytes: int, chunk_size_bytes: int
) -> bytes:
    file_contents_buffer = io.BytesIO()
    total_bytes_read = 0

    if hasattr(file, "chunks"):
        async for chunk in file.chunks(chunk_size_bytes):
            total_bytes_read += len(chunk)
            if total_bytes_read > max_size_bytes:
                file_contents_buffer.close()
                raise TextProcessingError(
                    f"Файл слишком большой. Максимальный размер: {max_size_bytes // (1024 * 1024)} МБ. "
                    f"Загружено более {total_bytes_read // (1024 * 1024)} МБ."
                )
            file_contents_buffer.write(chunk)
    else:
        while True:
            chunk = await file.read(chunk_size_bytes)
            if not chunk:
                break
            total_bytes_read += len(chunk)
            if total_bytes_read > max_size_bytes:
                file_contents_buffer.close()
                raise TextProcessingError(
                    f"Файл слишком большой. Максимальный размер: {max_size_bytes // (1024 * 1024)} МБ. "
                    f"Загружено более {total_bytes_read // (1024 * 1024)} МБ."
                )
            file_contents_buffer.write(chunk)

    if total_bytes_read == 0:
        file_contents_buffer.close()
        raise TextProcessingError("Загруженный файл пуст.")

    contents_bytes = file_contents_buffer.getvalue()
    file_contents_buffer.close()
    return contents_bytes


def decode_text_content(contents_bytes: bytes) -> str:
    try:
        text_content = contents_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_content = contents_bytes.decode("cp1251")
        except UnicodeDecodeError:
            raise TextProcessingError(
                "Не удалось декодировать файл. Пожалуйста, убедитесь, что файл в кодировке UTF-8 или CP1251."
            )

    if not text_content.strip():
        raise TextProcessingError(
            "Файл пустой или содержит только пробельные символы после декодирования."
        )
    return text_content


def calculate_tfidf_and_get_top_words(
    text_content: str, n_rows: int = N_ROWS_DEFAULT
) -> List[Dict[str, Any]]:
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text_content])

        feature_names = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_

        sum_of_all_tfidfs = vectors.sum()
        if sum_of_all_tfidfs == 0:
            if not feature_names.any():
                raise TextProcessingError(
                    "Не удалось извлечь слова из текста для анализа. Возможно, он состоит только из стоп-слов или слишком короткий."
                )
            tf_custom_values = [0.0] * len(feature_names)
        else:
            tf_custom_values = vectors.sum(axis=0).A1 / sum_of_all_tfidfs

        df = pd.DataFrame(
            {"TF": tf_custom_values, "IDF": idf_values}, index=feature_names
        )

        df = df.sort_values(by=["IDF", "TF"], ascending=[False, False])
        df_top_n = df.head(n_rows)

        output_results = []
        for word, row_data in df_top_n.iterrows():
            output_results.append(
                {"word": word, "tf": row_data["TF"], "idf": row_data["IDF"]}
            )
        return output_results
    except Exception as e:
        raise TextProcessingError(f"Ошибка при вычислении TF-IDF: {str(e)}")


async def process_uploaded_file_logic(
    file: Any,
    max_size_bytes: int,
    chunk_size_bytes: int,
    n_rows_output: int = N_ROWS_DEFAULT,
) -> List[Dict[str, Any]]:
    contents_bytes = await read_and_validate_file_content(
        file, max_size_bytes, chunk_size_bytes
    )
    text_content = decode_text_content(contents_bytes)
    results = calculate_tfidf_and_get_top_words(text_content, n_rows_output)
    return results
