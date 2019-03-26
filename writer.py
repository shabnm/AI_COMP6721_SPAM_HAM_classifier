class WriteToExternal:

    def write_to_model(self, i, word, word_count, probability):
        address = "C:\\Users\\admin\\PycharmProjects\\untitled\\model.txt"
        with open(address,"a") as f:
            f.write(word + " " + str(word_count) + " " + str(probability)+"\n")
            f.close()
