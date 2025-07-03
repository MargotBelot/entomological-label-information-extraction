# Import third-party libraries
from __future__ import annotations
import io
import os
import warnings
from google.cloud import vision

# Import the necessary module from the 'label_processing' module package
import label_processing.utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

class VisionApi:
    """
    Class for interacting with the Google Cloud Vision API for OCR tasks on images.
    """

    def __init__(self, path: str, image: bytes, credentials: str, encoding: str) -> None:
        """
        Initialize the VisionApi instance.

        Args:
            path (str): Path to the image file.
            image (bytes): Image content in bytes.
            credentials (str): Path to the credentials JSON file.
            encoding (str): Encoding for the result ('ascii' or 'utf8').
        """
        self.image = image
        self.path = path
        self.encoding = encoding
        self.credentials = credentials
        self.client = self._initialize_client(credentials)

    @staticmethod
    def _initialize_client(credentials: str) -> vision.ImageAnnotatorClient:
        """
        Initialize the Google Vision API client.

        Args:
            credentials (str): Path to the credentials JSON file.

        Returns:
            vision.ImageAnnotatorClient: Initialized Google Vision API client.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
        return vision.ImageAnnotatorClient()

    @staticmethod
    def read_image(path: str, credentials: str, encoding: str = 'utf8') -> VisionApi:
        """
        Read an image file and return an instance of the VisionApi class.

        Args:
            path (str): Path to the image file.
            credentials (str): Path to the credentials JSON file.
            encoding (str, optional): Encoding for the result ('ascii' or 'utf8'). Defaults to 'utf8'.

        Returns:
            VisionApi: Instance of the VisionApi class.
        """
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)

    def process_string(self, result_raw: str) -> str:
        """
        Process the Google Vision OCR output, replacing newlines with spaces and encoding as specified.

        Args:
            result_raw (str): Raw output string directly from Google Vision.

        Returns:
            str: Processed string.
        """
        processed = result_raw.replace('\n', ' ')
        if self.encoding == "ascii":
            processed = processed.encode("ascii", "ignore").decode()
        return processed

    def vision_ocr(self) -> dict[str, str]:
        """
        Perform the actual API call, handle errors, and return the processed transcription.

        Raises:
            Exception: Raises an exception if the API does not respond.

        Returns:
            dict[str, str]: Dictionary with the filename and the transcript.
        """
        vision_image = vision.Image(content=self.image)
        response = self.client.text_detection(image=vision_image)
        single_transcripts = response.text_annotations
        
        transcripts = [str(transcript.description) for transcript in single_transcripts]
        bounding_boxes = []

        for transcript in single_transcripts: 
            vertices = [{"word": f"({vertex.x},{vertex.y})"} for vertex in transcript.bounding_poly.vertices]
            bounding_boxes.append({transcript.description: vertices})

        if transcripts:
            transcript = self.process_string(transcripts[0])
        else:
            transcript = " "
        
        filename = os.path.basename(self.path)
        if response.error.message:
            raise Exception(
                f'{response.error.message}\nFor more info on error messages, '
                'check:  https://cloud.google.com/apis/design/errors'
            )
        
        entry = {'ID': filename, 'text': transcript, 'bounding_boxes': bounding_boxes}
        if label_processing.utils.check_text(entry["text"]): 
            entry = label_processing.utils.replace_nuri(entry)
        return entry