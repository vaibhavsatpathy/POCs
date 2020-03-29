package labda.DDICorpus.xml;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.List;

import labda.DDICorpus.utils.FileListFilter;

public class CollectionDDI {
	/**
     * Field _documentList.
     */
    private List<DocumentDDI> _documentList;
    public CollectionDDI() {
        super();
        this._documentList = new ArrayList<DocumentDDI>();
    }
    public CollectionDDI(String sDir) {
        super();
        this._documentList = new ArrayList<DocumentDDI>();
        loadCollection(sDir);
    }
    
    /**
     * Method getDocuments.Returns the contents of the collection in
     * an Array.  <p>Note:  Just in case the collection contents
     * are changing in another thread, we pass a 0-length Array of
     * the correct type into the API call.  This way we <i>know</i>
     * that the Array returned is of exactly the correct length.
     * 
     * @return this collection as an Array
     */
    public DocumentDDI[] getDocuments() {
    		DocumentDDI[] array = new DocumentDDI[0];
        return (DocumentDDI[]) this._documentList.toArray(array);
    }

    /**
     * Method getDocumentDDICount.
     * 
     * @return the size of this collection
     */
    public int getDocumentDDICount(
    ) {
        return this._documentList.size();
    }

    /**
     * Method getDocumentDDI.
     * 
     * @param index
     * @throws IndexOutOfBoundsException if the index
     * given is outside the bounds of the collection
     * @return the value of the Document at
     * the given index
     */
    public DocumentDDI getDocumentDDI(int index)
    throws IndexOutOfBoundsException {
        // check bounds for index
        if (index < 0 || index >= this._documentList.size()) {
            throw new IndexOutOfBoundsException("getDocumentDDI: Index value '" + index + "' not in range [0.." + (this._documentList.size() - 1) + "]");
        }
        
        return (DocumentDDI) _documentList.get(index);
    }
    /**
     * load the files in the folder sDir into the list of documents
     * @param sDir
     */
    private void loadCollection(String sDir) {
    	 try {
 			File oDir=new File(sDir);
 			if (!oDir.exists() || !oDir.isDirectory()) {
 				System.out.println("The folder " + sDir +" does not exist!!");
 			}
 			FilenameFilter select = new FileListFilter("", "xml");
 			File[] aFile=oDir.listFiles(select);
 			for (File oFile:aFile) {
 				//System.out.println(oFile.getAbsolutePath());
 				DocumentDDI newDoc=DocumentDDI.unmarshal(oFile.getAbsolutePath());
 				this._documentList.add(newDoc);
 			}
 			
 			
 	    } catch (Exception io ) {
 	        System.out.println( io.getMessage() );
 	    } 
    }
}
