package labda.DDICorpus.xml;

/**
 * Class Pair.
 * 	@author Isabel Segura-Bedmar isegura@inf.uc3m.es
 * @date 	5 Marzo 2015 
 */
public class Pair {


      //--------------------------/
     //- Class/Member Variables -/
    //--------------------------/

    public Pair(String _id, String _e1, String _e2, String _ddi, String _type) {
		super();
		this._id = _id;
		this._e1 = _e1;
		this._e2 = _e2;
		this._ddi = _ddi;
		this._type = _type;
	}

	/**
     * Field _id. id of the pair
     */
    private String _id;

    /**
     * Field _e1 for entity1
     */
    private String _e1;

    /**
     * Field _e2 for entity2
     */
    private String _e2;

    /**
     * Field _ddi value is true if the pair of entities involves an interaction, false eoc
     */
    private String _ddi;

    /**
     * Field _type: mechanism, advise, int, effect
     */
    private String _type;


      //----------------/
     //- Constructors -/
    //----------------/

    public Pair() {
        super();
    }


      //-----------/
     //- Methods -/
    //-----------/

    /**
     * Returns the value of field 'ddi'.
     * 
     * @return the value of field 'Ddi'.
     */
    public String getDdi(
    ) {
        return this._ddi;
    }

    /**
     * Returns the value of field 'e1'.
     * 
     * @return the value of field 'E1'.
     */
    public String getE1(
    ) {
        return this._e1;
    }

    /**
     * Returns the value of field 'e2'.
     * 
     * @return the value of field 'E2'.
     */
    public String getE2(
    ) {
        return this._e2;
    }

    /**
     * Returns the value of field 'id'.
     * 
     * @return the value of field 'Id'.
     */
    public String getId(
    ) {
        return this._id;
    }

    /**
     * Returns the value of field 'type'.
     * 
     * @return the value of field 'Type'.
     */
    public String getType(
    ) {
        return this._type;
    }

  
    /**
     * Sets the value of field 'ddi'.
     * 
     * @param ddi the value of field 'ddi'.
     */
    public void setDdi(
            final String ddi) {
        this._ddi = ddi;
    }

    /**
     * Sets the value of field 'e1'.
     * 
     * @param e1 the value of field 'e1'.
     */
    public void setE1(
            final String e1) {
        this._e1 = e1;
    }

    /**
     * Sets the value of field 'e2'.
     * 
     * @param e2 the value of field 'e2'.
     */
    public void setE2(
            final String e2) {
        this._e2 = e2;
    }

    /**
     * Sets the value of field 'id'.
     * 
     * @param id the value of field 'id'.
     */
    public void setId(
            final String id) {
        this._id = id;
    }

    /**
     * Sets the value of field 'type'.
     * 
     * @param type the value of field 'type'.
     */
    public void setType(
            final String type) {
        this._type = type;
    }

  public String toString() {
	  String str="\t"+this._id+"\t"+this._e1+"\t"+this._e2+"\t"+this._ddi;
	  if (this._type!=null) str+="\t"+this._type;
	  return str;
  }

}
