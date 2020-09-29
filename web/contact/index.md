---
layout: template1
comments: false
---


<!-- <div class="panel panel-default shadow1"> -->
<div class="panel panel-primary shadow1">
      <div class="panel-heading">
        <!-- <h4 class="text-primary">Contact</h4> -->
        <h1 class="panel-title">Contact</h1>
        <!-- <h3>Contact</h3> -->
        
        For any question or suggestion, please contact me.
        <br/>
        <br/>

      </div>
      
      
      <div class="panel-body">
            <form id="contactform" method="POST">
                  <div class="form-group">
                        <label for="name" class="control-label">Full name</label>
                        <input id="name" type="text" name="name" class="form-control">
                  </div>
                  <div class="form-group">
                        <label for="email" class="control-label">Email</label>
                        <input id="email" type="email" name="reply_to" class="form-control">
                  </div>
                  <div class="form-group">
                        <label for="msg" class="control-label">Message</label>
                        <textarea id="msg" name="message" class="form-control"></textarea>
                  </div>

                  <input type="text" name="_gotcha" style="display:none" />
                  <input type="submit" value="Send" class="btn btn-primary">
            </form>
      </div>
</div>

<script>
    var contactform =  document.getElementById('contactform');
    contactform.setAttribute('action', '//formspree.io/{{ site.profile.email }}');
</script>
