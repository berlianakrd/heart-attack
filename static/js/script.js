document.addEventListener('DOMContentLoaded', function(){
  
  const form = document.getElementById('predict-form');
  if(form){
    form.addEventListener('submit', function(e){
      // tampilkan loading sederhana
      const btn = form.querySelector('button[type="submit"]');
      if(btn){
        btn.disabled = true;
        btn.textContent = 'Memproses...';
      }
      
    });
  }

  document.querySelectorAll('input[type="number"]').forEach(function(inp){
    inp.addEventListener('input', function(){
      if(this.value && Number(this.value) < 0) this.value = '';
    });
  });
});