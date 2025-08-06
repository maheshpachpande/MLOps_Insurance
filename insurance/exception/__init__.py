




import os
import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)






# import sys

# def error_message_detail(error: Exception) -> str:
#     exc_type, exc_obj, exc_tb = sys.exc_info()
    
#     if exc_tb is not None:
#         file_name = exc_tb.tb_frame.f_code.co_filename
#         line_number = exc_tb.tb_lineno
#     else:
#         file_name = "Unknown File"
#         line_number = "Unknown Line"

#     return (
#         "\n" + "=" * 50 +
#         f"\nError occurred in ===>>> [ {file_name} ]"
#         f"\nline number [{line_number}]"
#         f"\nError Message: {str(error)}"
#         f"\n{'=' * 50}"
#     )

# class CustomException(Exception):
#     def __init__(self, message):
#         self.message = message
#         super().__init__(self.message)

    # def __str__(self):
    #     return self.error_message

# # Simulate an error
# if __name__ == "__main__":
#     try:
#         x = 1 / 0  # Intentional error
#     except Exception as e:
#         raise CustomException(e)
