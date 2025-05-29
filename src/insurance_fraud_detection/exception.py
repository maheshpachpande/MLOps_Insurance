
import sys #  for getting the file name and line number of the error

# this function is used to format the error message with details about the error
# and the location in the code where it occurred.
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # this will give us the traceback
    file_name=exc_tb.tb_frame.f_code.co_filename # this will give us the file name where the error occurred
    # this will give us the line number where the error occurred
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error)) # formatting the error message

    return error_message

# This class is used to create a custom exception that can be raised in the application.
class CustomException(Exception):
    def __init__(self,error_message,error_details:sys): # this will take in the error message and the error details
        super().__init__(error_message) # this will call the parent class constructor for Exception
        # this will format the error message with details about the error and the location in the code where it occurred
        self.error_message=error_message_detail(error_message,error_details) # this will give us the formatted error message

    def __str__(self):
        return self.error_message # this will return the formatted error message when the exception is raised
    # this will return the formatted error message when the exception is raised