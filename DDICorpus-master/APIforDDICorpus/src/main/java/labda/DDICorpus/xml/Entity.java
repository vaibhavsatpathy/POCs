package labda.DDICorpus.xml;

import java.util.List;
import java.util.ArrayList;


/**
 * Class Entity.
 * 
 * @author Isabel Segura-Bedmar isegura@inf.uc3m.es
 * @date 	5 Marzo 2015
 */
public class Entity {

	
      //--------------------------/
     //- Class/Member Variables -/
    //--------------------------/

    public Entity(String _id, String _text, String _type, String _charOffset) {
		super();
		this._id = _id;
		this._type = _type;
		this._text = _text;
		this.setCharOffset(_charOffset);

	}
    
    public Entity(String _id, String _text, String _charOffset) {
		super();
		this._id = _id;
		this._text = _text;
		this.setCharOffset(_charOffset);

	}

	/**
     * Field _id.
     */
    private String _id;

    
    /**
     * Field _charOffset: start and end positions in the sentence
     */
    private String _charOffset;
    
    private int start;
    private int end;
    
    

    /**
     * Field _type: drug, band, group, drug_n
     */
    private String _type;

    /**
     * Field _text: mention of the entity
     */
    private String _text;


    private List<Location> _locations;
      //----------------/
     //- Constructors -/
    //----------------/

    public Entity() {
        super();
    }


      //-----------/
     //- Methods -/
    //-----------/

    /**
     * Returns the value of field 'charOffset'.
     * 
     * @return the value of field 'CharOffset'.
     */
    public String getCharOffset() {
        return this._charOffset;
    }
    
    /**
     * Returns the value of field 'id'.
     * 
     * @return the value of field 'Id'.
     */
    public String getId() {
        return this._id;
    }

  
    public boolean equalsText(Entity obj) {
    		return (obj!=null && this.getText().equals(obj.getText()));
    }
    
    public boolean equalsOffSet(Entity obj) {
    		
		return (obj!=null && this.getCharOffset().equals(obj.getCharOffset()));
		
}

    /**
     * Returns the value of field 'text'.
     * 
     * @return the value of field 'Text'.
     */
    public String getText() {
        return this._text;
    }

    /**
     * Returns the value of field 'type'.
     * 
     * @return the value of field 'Type'.
     */
    public String getType() {
        return this._type;
    }

  

    /**
     * Sets the value of field 'charOffset'.
     * 
     * @param charOffset the value of field 'charOffset'.
     */
    public void setCharOffset(String charOffset) {
    		if (charOffset!=null) {
    	        this._charOffset = charOffset;

    			this._locations=new ArrayList<Location>();
    			int pos=-1;
    			while ((pos=charOffset.indexOf(";"))>-1) {
    				String firstToken=charOffset.substring(0,pos);
    				String sStart=firstToken.substring(0, firstToken.indexOf("-"));
    	    			String sEnd=firstToken.substring(firstToken.indexOf("-")+1);
    	    			Location loc=new Location(Integer.valueOf(sStart),Integer.valueOf(sEnd));
    	    			this._locations.add(loc);
    				charOffset=charOffset.substring(pos+1);
    			}
    			
	    		String sStart=charOffset.substring(0, charOffset.indexOf("-"));
	    		String sEnd=charOffset.substring(charOffset.indexOf("-")+1);
	    		
	    		Location loc=new Location(Integer.valueOf(sStart),Integer.valueOf(sEnd));
    			this._locations.add(loc);
    		}
    }

    /**
     * Sets the value of field 'id'.
     * 
     * @param id the value of field 'id'.
     */
    public void setId(String id) {
        this._id = id;
    }

   

    /**
     * Sets the value of field 'text'.
     * 
     * @param text the value of field 'text'.
     */
    public void setText(String text) {
        this._text = text;
    }

    /**
     * Sets the value of field 'type'.
     * 
     * @param type the value of field 'type'.
     */
    public void setType(String type) {
        this._type = type;
    }

    public String toString() {
  	  String str="\t"+this._id+"\t"+this._charOffset+"\t"+this._text+"\t"+this._type;
  	  if (this._locations!=null) {
  		  for (Location loc:this._locations) {
  		  	  str+="\n\t\t"+loc._start+"-"+loc._end; 
  		  }
  	  }
  	  return str;
    }


	public List<Location> getLocations() {
		return _locations;
	}


	public void setLocations(List<Location> _locations) {
		this._locations = _locations;
	}
	
	private int[] getPositions() {
		String offSet=this.getCharOffset();
		int pos=offSet.indexOf(";");
		if (pos>-1) {
			String aData[]=offSet.split(";");
			offSet=aData[0];
		}
		String aData[]=offSet.split("-");

		int[] iData=new int[2];
		iData[0]=Integer.parseInt(aData[0]);
		iData[1]=Integer.parseInt(aData[1]);
		return iData;
	}
	public boolean contains(Entity obj) {
		if (obj==null || obj.getText()==null) return false;
		int[] positions=obj.getPositions();
		int[] thisPos=this.getPositions();
		if (positions[0]<=thisPos[0] && positions[0]<=thisPos[1]) return true;
		else return false;
	}

	public boolean overlap(Entity obj) {
		if (this.equalsOffSet(obj) ||this.contains(obj) ||obj.contains(this)) return true;
		int[] positions=obj.getPositions();
		int[] thisPos=this.getPositions();
		
		int start1=positions[0];
		int end1=positions[1];
		
		int start=thisPos[0];
		int end=thisPos[1];
		
		if (start1>end || start>end1) return false;
		else return true;
		
		
	}
    
}
