from file_path import File_Path

class WriteToExternal:

    def write_to_model(self, i, word, ham_word_count, ham_probability, spam_word_count, spam_probability):
        address = File_Path.MODEL
        with open(address, "a") as f:
            f.write("{}  {}  {}  {}  {}  {}\n".format(str(i), word, str(ham_word_count),str(ham_probability),
                                                      str(spam_word_count), str(spam_probability)))
            f.close()
