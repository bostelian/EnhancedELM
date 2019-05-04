class VerbosityManager:
    def __init__(self, verbose = False, class_name = None):
        self.verbose = verbose
        self.call_stack = []
        self.class_name = class_name
        
    def begin(self, function_name = None):
        if(self.verbose):
            print("Begin:{0}.{1}".format(self.class_name,function_name))
            self.call_stack.append(function_name)
    
    def end(self):
         if(self.verbose):
            print("End:{0}.{1}".format(self.class_name, self.call_stack.pop()))